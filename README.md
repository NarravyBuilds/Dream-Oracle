---
title: Dream Oracle
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: false
---

# Dream Oracle

Traumdeutungs-Bot mit RAG-basierter Interpretation. Nutzt eine Wissensbasis aus Traumsymbolen, anonymisierten Traumberichten und psychologischen Perspektiven, um vorsichtige, nachvollziehbare Deutungen zu generieren.

## Architektur

```
Traumtext → Embeddings → Wissensbasis durchsuchen (RAG) → LLM-Antwort generieren
```

- **Embeddings:** Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Vektordatenbank:** Qdrant (lokal)
- **LLM:** Meta Llama 3.2 via HF Inference API
- **UI:** Gradio

## Lokales Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
