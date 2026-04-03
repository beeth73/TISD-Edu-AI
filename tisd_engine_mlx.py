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

# --- THE ABSOLUTE PATH FIX ---
# This looks for the subfolder 'chroma_db' inside 'vectorstore'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if "notebooks" in BASE_DIR:
    BASE_DIR = os.path.dirname(BASE_DIR)

# Force the path to where your ls -R showed the data
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_db")
COLLECTION_NAME = "tisd_knowledge"

class TISDEngine:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model, self.tokenizer, self.embedder, self.collection = None, None, None, None
        self.sampler = make_sampler(temp=0.0)

    def load(self):
        t0 = time.time()
        if self.verbose: print(f"🔍 Checking subfolder: {VECTORSTORE_DIR}")
        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        
        # Diagnostic print
        existing_collections = client.list_collections()
        existing_names = [c.name for c in existing_collections]
        if self.verbose: print(f"📚 Collections found: {existing_names}")

        if not existing_names:
            raise RuntimeError(f"❌ ERROR: No collections in {VECTORSTORE_DIR}. Check your rebuild script paths!")
        
        active_name = COLLECTION_NAME if COLLECTION_NAME in existing_names else existing_names[0]
        self.collection = client.get_collection(active_name)
        
        self.model, self.tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
        
        if self.verbose:
            print(f"✅ Engine Loaded | RAM: {round(psutil.virtual_memory().used / 1e9, 2)}GB")

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
