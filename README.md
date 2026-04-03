
# 🎒 TISD: The Intelligent Student Desk
**Local-First, Native-Performance RAG Pipeline for Grade 1–10 Academic Assistance.**

TISD (The Intelligent Student Desk) is a privacy-focused, high-performance AI tutor designed specifically for Apple Silicon. By leveraging the **M4 Unified Memory Architecture** and the **MLX framework**, TISD provides near-instantaneous, grounded academic support without ever sending data to the cloud.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![MLX](https://img.shields.io/badge/Framework-MLX-red.svg)
![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4-gold.svg)
![Accuracy](https://img.shields.io/badge/Semantic%20Accuracy-81.1%25-green.svg)

---

## 🚀 The Vision
Most educational AI tools are wrappers for cloud-based LLMs that are prone to hallucinations and privacy leaks. TISD is an engineering-first solution that:
1.  **Runs 100% Locally:** Zero data leaves the student's device.
2.  **Eliminates Hallucinations:** Answers are strictly grounded in verified textbooks (NCERT) and encyclopedias (DK, Britannica).
3.  **M4 Optimized:** Native Apple Silicon execution for elite speed and power efficiency.

## 🧠 Technical Stack & Architecture
- **Core LLM:** `Microsoft Phi-3-Mini-4k-Instruct` (3.8B Parameters)
- **Inference Engine:** `MLX-LM` (Apple's Native Machine Learning Framework)
- **Vector Database:** `ChromaDB` (Metadata-indexed semantic retrieval)
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2`
- **UI:** FastAPI backend with a Cartier-inspired high-fidelity Frontend (Tailwind CSS/JS)

### The Retrieval-Augmented Generation (RAG) Flow:
1.  **Ingest:** Raw PDFs are cleaned using Regex and chunked into overlapping blocks.
2.  **Index:** Chunks are tagged with metadata (`class_level`, `subject`, `source_type`) and stored in ChromaDB.
3.  **Retrieve:** User queries are semantically matched against the vector store.
4.  **Synthesize:** The Phi-3 model performs **Extractive Reasoning** to generate a pedagogical response.

---

## 📊 Benchmarks (Apple Silicon M4)
*Tested on MacBook Air M4, 16GB Unified Memory.*

| Metric | Performance |
| :--- | :--- |
| **Prompt Processing Speed** | 223.99 Tokens/sec |
| **Generation Speed** | 41.42 Tokens/sec |
| **Average Semantic Accuracy** | 81.11% (100-sample Stress Test) |
| **Peak Memory Footprint** | ~9.99 GB |
| **Average Latency** | 1.72 Seconds |

---

## 📊 Data & Compute Scale
TISD isn't just a "toy" chatbot; it is a high-density retrieval system.
Dataset Depth: Processed 4,656 pages of academic text, including the full NCERT Grade 1–4 curriculum and high-resolution encyclopedias.
Granularity: The pipeline successfully generated and indexed 6,422 semantic chunks.
Mathematical Complexity: The system manages a vector space of 6,422 x 384 dimensions. Every query involves a real-time similarity calculation across 2.4 million data points, executed in milliseconds on the M4 GPU.
Resilience: The ingestion engine is built to handle "real-world" data, successfully recovering from stream errors and EOF (End Of File) markers in legacy PDF formats without crashing the pipeline.

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
conda create -n tisd python=3.11 -y
conda activate tisd
pip install mlx-lm chromadb sentence-transformers fastapi uvicorn psutil python-dotenv pypdf
```

### 2. Knowledge Base Construction
Place your academic PDFs in `data/raw/` and execute the automated pipeline:
```bash
chmod +x rebuild_tisd.sh
./rebuild_tisd.sh
```

### 3. Localhost Deployment
Launch the "Tara" AI Teacher interface:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000` for the Cartier-inspired UI.

---

## 🛡️ Evaluation & Reliability
TISD includes a **SOTA Stress-Testing Suite** (`08_sota_eval.ipynb`) that:
- Runs 100 diverse and obscure academic questions.
- Implements a **Thermal-Aware Protocol** (45s cooldown every 25 queries) for fanless MacBooks.
- Calculates **Semantic Cosine Similarity** against ground-truth datasets.

## 🗺️ Roadmap
- [ ] **Cross-Encoder Reranking:** Moving from 81% to 90%+ accuracy.
- [ ] **Multimodal Latents:** Unified CLIP-based search for textbook diagrams.
- [ ] **Socratic Mode:** LoRA adapter for inquiry-based learning.
- [ ] **10,000 Sample Benchmarking:** Large-scale academic validation.

---
**Author:** beeth73  
**License:** MIT  
***"The tire marks on the track are proof that engineering happened here."***
---
