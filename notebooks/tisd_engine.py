# ~/tisd/notebooks/tisd_engine.py
import torch
import faiss
import json
import os
import time
import warnings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, HTML

# Global variables for the engine
device = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map={"": device}
)
model = PeftModel.from_pretrained(base_model, "../models/tisd-tinyllama-lora")
index = faiss.read_index("../vectorstore/faiss_index.bin")
with open("../vectorstore/chunk_metadata.json", "r") as f:
    document_chunks = json.load(f)

def chat_with_tisd(question, top_k=3):
    # Paste your logic here exactly as it was in Notebook 05
    # (Just ensure imports like 'time' are present in this file)
    start_time = time.time()
    
    # 1. Retrieve context
    question_vector = encoder.encode([question], normalize_embeddings=True)
    distances, indices = index.search(question_vector, top_k)
    
    retrieved_contexts = [document_chunks[indices[0][i]]['chunk_text'] for i in range(top_k)]
    combined_context = " ".join(retrieved_contexts)
    
    # 2. TWEAKED PROMPT: Added the "I don't know" safety valve
    # In tisd_engine.py
    system_prompt = "You are a teacher for grade 1-4 students."
    
    # We give it 2 "perfect" examples (Few-Shot)
    few_shot = (
        "Context: The Sun is a star that gives us heat and light. \n"
        "Question: What is the Sun? \n"
        "Answer: The Sun is a star that gives us light and heat.\n\n"
        "Context: Plants need water, air, and sunlight to grow. \n"
        "Question: What do plants need? \n"
        "Answer: Plants need water, air, and sunlight.\n\n"
    )
    
    full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{few_shot}Context: {combined_context}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # 3. Generate Answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,   # Increased from 150 so it doesn't get cut off
            temperature=0.1,      # Lowered to 0.1 to make it strictly follow our instructions
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    # 4. Clean up the output string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("<|assistant|>\n")[-1].strip()
    
    # Failsafe cleanups just in case TinyLlama acts stubborn
    if answer.startswith("Answer:"):
        answer = answer.replace("Answer:", "").strip()
        
    elapsed_time = time.time() - start_time
    
    return answer, combined_context, elapsed_time

def post_process_answer(answer):
    # Brutal cleaning
    forbidden_phrases = ["The context mentions", "The question asks", "To summarize", "Question:"]
    for phrase in forbidden_phrases:
        answer = answer.replace(phrase, "")
    return answer.strip()