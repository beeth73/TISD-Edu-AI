import torch
import json
import time
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

warnings.filterwarnings('ignore')

# --- Initialization ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.bfloat16, 
    device_map={"": device}
)
model = PeftModel.from_pretrained(base_model, "../models/tisd-tinyllama-lora")

client = chromadb.PersistentClient(path="../vectorstore/chroma_db")
collection = client.get_collection(name="tisd_knowledge_base")

def post_process_answer(answer):
    """Aggressive cleanup to stop any conversational filler."""
    # If the model tries to ask a question back, cut it off
    if "?" in answer:
        answer = answer.split("?")[0]
    
    forbidden = ["The context says", "The context states", "Based on the text", "Yes, please", "Question:", "Answer:"]
    for phrase in forbidden:
        # Case insensitive replace
        answer = answer.replace(phrase, "").replace(phrase.lower(), "")
        
    return answer.strip()

def chat_with_tisd(question, top_k=2, class_filter=None):
    start_time = time.time()
    
    # 1. PURE SEMANTIC RETRIEVAL (Low noise)
    where_clause = {"class": str(class_filter)} if class_filter else None
    results = collection.query(query_texts=[question], n_results=top_k, where=where_clause)
    
    # Safely get chunks and truncate to prevent context poisoning
    retrieved_contexts = results['documents'][0]
    combined_context = " ".join(retrieved_contexts)[:800] 
    
    # 2. STRICT PROMPT + CHATML TAGS (The Golden Mean)
    system_prompt = (
        "You are a strict data extractor. "
        "Answer the Question using ONLY the exact facts from the Information provided. "
        "Keep your answer under 2 sentences. Do not ask questions."
    )
    
    full_prompt = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\nInformation: {combined_context}\n\nQuestion: {question}</s>\n"
        f"<|assistant|>\n"
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # 3. DETERMINISTIC GENERATION
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,   
            temperature=0.01,    
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    # 4. PARSE OUTPUT
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only what comes after the assistant tag
    raw_answer = raw_output.split("<|assistant|>\n")[-1].strip()
    
    final_answer = post_process_answer(raw_answer)
    
    # SOTA Failsafe: If the model generated nothing, or a single weird word
    if len(final_answer) < 5 or "Yes" in final_answer:
        final_answer = "I'm sorry, that specific answer isn't in my textbooks yet!"
    
    return final_answer, combined_context, time.time() - start_time