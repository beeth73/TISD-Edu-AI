# ~/tisd/notebooks/tisd_engine_mlx.py
import time
import chromadb
import mlx.core as mx
from mlx_lm import load, generate

# --- INITIALIZATION (Native M4 Architecture) ---
# We use the community-provided optimized model
model_path = "mlx-community/Phi-3-mini-4k-instruct-4bit"
model, tokenizer = load(model_path)

# Chroma Setup
client = chromadb.PersistentClient(path="../vectorstore/chroma_db")
collection = client.get_collection(name="tisd_knowledge_base")

def post_process_answer(answer):
    forbidden = ["The context says", "Question:", "Answer:", "<|assistant|>"]
    for phrase in forbidden:
        answer = answer.replace(phrase, "").replace(phrase.lower(), "")
    return answer.strip()

def chat_with_tisd_mlx(question, top_k=3, class_filter=None):
    start_time = time.time()
    
    # 1. RETRIEVAL (Standard Semantic)
    where_clause = {"class": str(class_filter)} if class_filter else None
    results = collection.query(query_texts=[question], n_results=top_k, where=where_clause)
    combined_context = " ".join(results['documents'][0])
    
    # 2. PROMPT (Phi-3 MLX Native Format)
    prompt = f"<|system|>\nYou are a teacher. Use the Information to answer the Question.\n<|end|>\n<|user|>\nInformation: {combined_context}\n\nQuestion: {question}\n<|end|>\n<|assistant|>\n"
    
    # 3. NATIVE MLX GENERATION
    # MLX is much faster on M4 because it doesn't use the 'transformers' overhead
    answer = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, 
        temp=0.1
    )
    
    final_answer = post_process_answer(answer)
    return final_answer, combined_context, time.time() - start_time