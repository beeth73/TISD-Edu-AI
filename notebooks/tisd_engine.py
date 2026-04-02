import torch
import json
import time
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
    forbidden = ["The context mentions", "The question asks", "To summarize", "Question:", "Answer:"]
    for phrase in forbidden:
        answer = answer.replace(phrase, "")
    return answer.strip()

def chat_with_tisd(question, top_k=2, class_filter=None):
    start_time = time.time()
    
    # 1. PURE SEMANTIC RETRIEVAL (ChromaDB)
    # Vectors understand "Meaning", not just keywords.
    where_clause = {"class": str(class_filter)} if class_filter else None
    
    results = collection.query(
        query_texts=[question], 
        n_results=top_k, 
        where=where_clause
    )
    
    # Get the top chunks
    retrieved_contexts = results['documents'][0]
    
    # Crucial: Truncate context to prevent overloading the 1.1B model
    combined_context = " ".join(retrieved_contexts)[:1000] # Limit to ~150 words
    
    # 2. THE "STRICT" PROMPT
    # No few-shot this time. Just extreme clarity.
    system_prompt = (
        "You are TISD, a teacher. "
        "Read the Context below. Answer the Question using ONLY the facts from the Context. "
        "Keep it to one sentence."
    )
    
    full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\nContext: {combined_context}\nQuestion: {question}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # 3. GENERATION
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, # Shorter output = less chance to hallucinate
            temperature=0.1,    # Extremely focused
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>\n")[-1].strip()
    final_answer = post_process_answer(raw_answer)
    
    return final_answer, combined_context, time.time() - start_time