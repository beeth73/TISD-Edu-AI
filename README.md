# TISD: The Intelligent Student Desk
A privacy-first, local RAG-enabled AI tutor for K-4 students. 

TISD provides an "open-book" AI experience where all answers are grounded strictly in NCERT-curriculum textbooks. It runs entirely on-device using Apple Silicon, ensuring student data privacy and zero cloud latency.

## Key Features
- **Privacy-First:** Local-only inference using Apple MPS (Metal Performance Shaders).
- **Grounded AI:** Retrieval-Augmented Generation (RAG) prevents hallucinations.
- **Multimodal:** Integrated textbook retrieval + YouTube learning resources.
- **Teacher-Persona:** Fine-tuned TinyLlama SLM adapted to explain concepts simply.

## Architecture
- **Retrieval:** Hybrid Search (ChromaDB + BM25) for high-accuracy textbook lookups.
- **Generator:** TinyLlama-1.1B fine-tuned with LoRA (Low-Rank Adaptation).
- **Safety:** Automatic Flesch-Kincaid readability scoring ensures output is grade-appropriate.

## Setup
```bash
conda create -n tisd python=3.11 -y
conda activate tisd
pip install -r requirements.txt