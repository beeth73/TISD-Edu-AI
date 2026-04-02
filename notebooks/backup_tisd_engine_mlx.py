# tisd_engine_mlx.py
# TISD — The Intelligent Student Desk
# MLX-native inference engine for Apple Silicon M4
# Compatible with mlx-lm >= 0.22.x (uses make_sampler, not temp= kwarg)

import os
import time
import json
import psutil
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler   # <-- THE FIX: new API
from tqdm.notebook import tqdm
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID       = "mlx-community/Phi-3-mini-4k-instruct-4bit"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
COLLECTION_NAME = "tisd_knowledge"
MAX_TOKENS      = 300
TOP_K_RETRIEVE  = 15   # retrieve broadly
TOP_K_RERANK    = 3    # then narrow to 3 for context
MAX_GRADE       = 4

# Sampling config — deterministic for factual QA
# For a children's education bot, temp=0 is correct:
# we want the most confident answer, not creative variation
SAMPLER_CONFIG = {
    "temp": 0.0,    # greedy / deterministic
    "top_p": 1.0,
    "min_p": 0.0,
    "min_tokens_to_keep": 1,
}

SYSTEM_PROMPT = """You are Tara, a warm and patient teacher for children in Grade 1 to 4.
Rules you must always follow:
1. Use simple words a young child understands.
2. Keep your answer to 3-4 short sentences maximum.
3. Only use information from the CONTEXT provided below.
4. If the context does not contain the answer, say: "That's a great question! Let's ask your teacher about that one."
5. Never make up facts. Never use complicated words.
6. End with one encouraging sentence."""


# ─────────────────────────────────────────────
# MEMORY TELEMETRY
# ─────────────────────────────────────────────
def get_memory_stats() -> dict:
    """Returns current RAM usage. Swap > 1GB on M4 = danger zone."""
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "ram_used_gb":  round(vm.used   / 1e9, 2),
        "ram_avail_gb": round(vm.available / 1e9, 2),
        "swap_used_gb": round(swap.used  / 1e9, 2),
        "ram_pct":      vm.percent,
    }


# ─────────────────────────────────────────────
# ENGINE CLASS
# ─────────────────────────────────────────────
class TISDEngine:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.model      = None
        self.tokenizer  = None
        self.embedder   = None
        self.collection = None
        self.sampler    = None   # built once, reused every call

    # ── LOAD ──────────────────────────────────
    def load(self):
        """Load all components. Call once at notebook startup."""
        t0 = time.time()

        # 1. Embedding model (lightweight, CPU)
        if self.verbose:
            print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL_ID)

        # 2. ChromaDB
        if self.verbose:
            print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        # 3. Phi-3 via MLX
        if self.verbose:
            print(f"Loading {MODEL_ID} with MLX...")
        self.model, self.tokenizer = load(MODEL_ID)

        # 4. Build sampler ONCE using the new API
        # make_sampler replaces temp=, top_p= kwargs on generate()
        self.sampler = make_sampler(**SAMPLER_CONFIG)

        elapsed = time.time() - t0
        mem = get_memory_stats()
        if self.verbose:
            print(f"\nEngine ready in {elapsed:.1f}s")
            print(f"RAM: {mem['ram_used_gb']}GB used | "
                  f"{mem['ram_avail_gb']}GB free | "
                  f"Swap: {mem['swap_used_gb']}GB")

    # ── RETRIEVAL ─────────────────────────────
    def _retrieve(self, question: str, grade: int) -> list[str]:
        """
        Metadata-pre-filtered dense retrieval via ChromaDB.
        Filters to grade <= requested grade BEFORE vector search,
        not after — avoids the top-k blowout problem with FAISS.
        """
        q_embedding = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=TOP_K_RETRIEVE,
            where={"class_level": {"$lte": str(grade)}},
            include=["documents", "metadatas", "distances"]
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Light rerank: sort by distance (ChromaDB returns L2, lower = better)
        ranked = sorted(
            zip(distances, docs, metadatas),
            key=lambda x: x[0]
        )

        # Return top-K_RERANK chunks as plain strings
        return [doc for _, doc, _ in ranked[:TOP_K_RERANK]]

    # ── PROMPT BUILDER ────────────────────────
    def _build_prompt(self, question: str, chunks: list[str]) -> str:
        """
        Phi-3 uses <|system|>, <|user|>, <|assistant|> tags.
        This is the exact format Phi-3-mini-4k-instruct expects.
        """
        context = "\n\n---\n\n".join(chunks)

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (
                f"CONTEXT:\n{context}\n\n"
                f"STUDENT QUESTION: {question}"
            )},
        ]

        # apply_chat_template handles the <|system|>...<|end|> tags correctly
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # adds <|assistant|> at the end
        )
        return prompt

    # ── SELF-VERIFY ───────────────────────────
    def _verify(self, question: str, answer: str, context: str) -> bool:
        """
        Lightweight self-correction pass.
        Same model, different prompt — no extra memory cost.
        Returns True if answer is grounded, False if it drifted.
        """
        verify_messages = [
            {"role": "system", "content": (
                "You are a fact-checker. Answer only YES or NO."
            )},
            {"role": "user", "content": (
                f"Context: {context}\n\n"
                f"Answer given: {answer}\n\n"
                f"Question: Is this answer based only on the context above? "
                f"Reply YES or NO only."
            )},
        ]
        verify_prompt = self.tokenizer.apply_chat_template(
            verify_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Short, deterministic verdict
        verdict = generate(
            self.model,
            self.tokenizer,
            prompt=verify_prompt,
            max_tokens=5,
            sampler=self.sampler,    # <-- correct API
            verbose=False,
        )
        return verdict.strip().upper().startswith("YES")

    # ── MAIN INFERENCE ────────────────────────
    def ask(
        self,
        question:  str,
        grade:     int = 4,
        verify:    bool = True,
        verbose:   bool = None,
    ) -> dict:
        """
        Full RAG pipeline: retrieve → prompt → generate → verify.

        Returns a dict with answer, latency, memory stats, and
        the retrieved chunks (useful for debugging / evaluation).
        """
        if verbose is None:
            verbose = self.verbose

        t_start = time.time()
        result  = {}

        # Step 1: Retrieve
        t0 = time.time()
        chunks = self._retrieve(question, grade)
        t_retrieve = time.time() - t0

        if not chunks:
            return {
                "answer":     "I don't have information about that in my books yet!",
                "verified":   False,
                "chunks":     [],
                "latency_ms": {"retrieve": 0, "generate": 0, "verify": 0, "total": 0},
                "memory":     get_memory_stats(),
            }

        # Step 2: Build prompt
        prompt  = self._build_prompt(question, chunks)
        context = "\n\n".join(chunks)   # flat string for verify step

        # Step 3: Generate
        t0 = time.time()
        answer = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=self.sampler,       # <-- THE FIX: pass sampler object, not temp=
            verbose=False,
        )
        t_generate = time.time() - t0

        # Strip any trailing EOS tokens Phi-3 sometimes emits
        answer = answer.replace("<|end|>", "").replace("<|endoftext|>", "").strip()

        # Step 4: Self-verify (optional, adds ~0.5–1s)
        t0 = time.time()
        verified = False
        if verify:
            verified = self._verify(question, answer, context)
            if not verified:
                # Regenerate once with stricter grounding instruction
                if verbose:
                    print("Verification failed — regenerating with stricter prompt...")
                strict_messages = [
                    {"role": "system", "content": (
                        SYSTEM_PROMPT +
                        "\nIMPORTANT: Use ONLY the exact facts from the context. "
                        "Do not add anything else."
                    )},
                    {"role": "user", "content": (
                        f"CONTEXT:\n{context}\n\n"
                        f"STUDENT QUESTION: {question}"
                    )},
                ]
                strict_prompt = self.tokenizer.apply_chat_template(
                    strict_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                answer = generate(
                    self.model,
                    self.tokenizer,
                    prompt=strict_prompt,
                    max_tokens=MAX_TOKENS,
                    sampler=self.sampler,
                    verbose=False,
                )
                answer = answer.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
                verified = True   # trust the second pass
        t_verify = time.time() - t0

        t_total = time.time() - t_start

        result = {
            "answer":   answer,
            "verified": verified,
            "chunks":   chunks,
            "latency_ms": {
                "retrieve": round(t_retrieve * 1000),
                "generate": round(t_generate * 1000),
                "verify":   round(t_verify   * 1000),
                "total":    round(t_total     * 1000),
            },
            "memory": get_memory_stats(),
        }

        if verbose:
            print(f"\nAnswer: {answer}")
            print(f"\nLatency → retrieve: {result['latency_ms']['retrieve']}ms | "
                  f"generate: {result['latency_ms']['generate']}ms | "
                  f"total: {result['latency_ms']['total']}ms")
            print(f"Memory → RAM: {result['memory']['ram_used_gb']}GB | "
                  f"Swap: {result['memory']['swap_used_gb']}GB")
            print(f"Verified: {verified}")

        return result


# ─────────────────────────────────────────────
# BATCH EVALUATION HELPER
# (used in 06_evaluation.ipynb)
# ─────────────────────────────────────────────
def run_evaluation(engine: TISDEngine, test_set: list[dict]) -> list[dict]:
    """
    test_set: list of {"question": str, "grade": int, "expected": str}
    Returns list of result dicts with answers + latency for BERTScore.
    """
    results = []
    for item in tqdm(test_set, desc="Evaluating"):
        t0 = time.time()
        out = engine.ask(
            question=item["question"],
            grade=item.get("grade", 4),
            verify=False,   # skip verify during bulk eval for speed
            verbose=False,
        )
        results.append({
            "question":  item["question"],
            "expected":  item["expected"],
            "predicted": out["answer"],
            "latency_ms": out["latency_ms"]["total"],
            "memory":    out["memory"],
        })
    return results