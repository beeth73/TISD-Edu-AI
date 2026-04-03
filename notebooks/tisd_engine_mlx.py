# Run this in a Jupyter cell to overwrite ~/tisd/tisd_engine_mlx.py
import os
import time
import psutil
import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class TISDEngine:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model, self.tokenizer, self.embedder, self.collection = None, None, None, None
        self.sampler = make_sampler(temp=0.0)

    def _find_vectorstore(self):
        # We check the two places your 'ls -R' showed data
        possible_paths = [
            os.path.expanduser("~/tisd/vectorstore/chroma_db"),
            os.path.expanduser("~/tisd/vectorstore")
        ]
        for p in possible_paths:
            if os.path.exists(os.path.join(p, "chroma.sqlite3")):
                return p
        return possible_paths[0] # Fallback

    def load(self):
        t0 = time.time()
        db_path = self._find_vectorstore()
        
        if self.verbose: 
            print(f"🔍 Engine looking for data in: {db_path}")
        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=db_path)
        
        # Diagnostic: What is actually in this DB?
        existing_collections = client.list_collections()
        if not existing_collections:
            raise RuntimeError(f"❌ Empty Database found at {db_path}. Run rebuild_tisd.sh again!")
            
        # Pick the collection (prioritize 'tisd_knowledge')
        names = [c.name for c in existing_collections]
        active_name = "tisd_knowledge" if "tisd_knowledge" in names else names[0]
             
        self.collection = client.get_collection(active_name)
        self.model, self.tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
        
        if self.verbose:
            print(f"✅ Engine v3.3 Ready | Collection: {active_name} | RAM: {round(psutil.virtual_memory().used / 1e9, 2)}GB")

    def clean_text(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        text = text.replace('bys', 'by')
        text = re.sub(r'Chapter \d+\.indd \d+', '', text)
        text = re.sub(r'Reprint \d+-\d+', '', text)
        text = re.sub(r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _retrieve(self, question, grade):
        q_emb = self.embedder.encode(question).tolist()
        res = self.collection.query(
            query_embeddings=[q_emb], 
            n_results=3, 
            where={"class_level": {"$lte": int(grade)}}
        )
        return res["documents"][0], res["metadatas"][0]

    def ask(self, question, grade=4, verbose=None):
        if verbose is None: verbose = self.verbose
        chunks, metas = self._retrieve(question, grade)
        context = " ".join([self.clean_text(c) for c in chunks])
        sources = [m.get("source", "Reference") for m in metas]
        
        messages = [
            {"role": "system", "content": "You are Tara, a teacher. Answer concisely using ONLY the facts in the context."},
            {"role": "user", "content": f"CONTEXT: {context}\\n\\nQUESTION: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=250, sampler=self.sampler, verbose=verbose)
        return self.clean_text(answer), list(set(sources))
