<div align="center">

# 🌙 Narravy Dream Oracle

**Erzähl mir deinen Traum und ich helfe dir, ihn zu verstehen.**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/OrangeDev/Narravy-Dream-Oracle)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.9-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-C4A882)](LICENSE)

<br>

*Ein KI-gestützter Traumdeutungs-Assistent, der Symbolik, semantische Suche und Sprachmodelle verbindet, um Träume einfühlsam zu interpretieren.*

<br>

<img src="https://img.shields.io/badge/-%E2%9C%A6%20AI%20%E2%80%A2%20NLP%20%E2%80%A2%20PYTHON%20%E2%9C%A6-1A1714?style=for-the-badge&labelColor=1A1714" alt="aesthetic">

</div>

---

## ✦ Was ist Dream Oracle?

Dream Oracle ist ein Traumdeutungs-Bot, der **RAG** (Retrieval-Augmented Generation) nutzt, um Träume kontextbewusst zu interpretieren. Anstatt generische Antworten zu geben, durchsucht er eine kuratierte Wissensbasis aus Traumsymbolen und realen Traumberichten, um eine persönliche und mehrschichtige Deutung zu liefern.

> *"Die träumende Person ist die beste Expertin für ihre eigenen Träume."*

Dream Oracle behauptet nie, die eine wahre Bedeutung zu kennen — er bietet Perspektiven an, keine Diagnosen.

---

## ✦ Features

🔮 **Symbolerkennung**
Automatische Identifikation von Traumsymbolen mit semantischer Ähnlichkeitssuche

📚 **RAG-Pipeline**
Kontextanreicherung durch eine Vektordatenbank mit Symbolen und echten Traumberichten

🏷️ **Motiv-Badges**
Visuelle Tags der erkannten Symbole über jeder Deutung

🛡️ **Safety Filter**
Erkennung sensibler Inhalte mit Verweis auf professionelle Hilfsangebote

🌙 **Dark Aesthetic**
Handgestaltetes UI mit Cormorant Garamond, Gold-Akzenten und cinematischer Atmosphäre

📱 **Mobile-Ready**
Optimiert für Smartphone mit dediziertem Submit-Button

---

## ✦ Architektur

```
Traum-Text
    │
    ├──→ Sentence Transformer (MiniLM-L12-v2)
    │         │
    │         ├──→ Qdrant: Symbole durchsuchen
    │         └──→ Qdrant: Ähnliche Berichte finden
    │
    ├──→ Prompt Builder (Template + Kontext)
    │
    └──→ LLM (Llama 3.2 1B via HF Inference API)
              │
              └──→ Deutung + Motiv-Badges
```

---

## ✦ Tech Stack

| Komponente | Technologie |
|---|---|
| **Frontend** | Gradio 6.9 mit Custom CSS |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Vektordatenbank** | Qdrant (lokal) |
| **LLM** | Meta Llama 3.2 1B Instruct |
| **API** | Hugging Face Inference Router |
| **Hosting** | Hugging Face Spaces |

---

## ✦ Lokale Installation

```bash
# Repository klonen
git clone https://github.com/NarravyBuilds/Dream-Oracle.git
cd Dream-Oracle

# Dependencies installieren
pip install -r requirements.txt

# HF Token setzen (für LLM-Zugriff)
export HF_TOKEN="hf_dein_token_hier"

# Starten
python app.py
```

Die App läuft dann auf `http://localhost:7860`.

---

## ✦ Datenstruktur

```
Dream-Oracle/
├── app.py                  # Hauptanwendung
├── requirements.txt        # Python-Abhängigkeiten
├── app/
│   └── prompts/
│       └── interpret_prompt.txt   # LLM Prompt-Template
├── data/
│   ├── symbols.json        # Traumsymbol-Datenbank
│   └── dream_reports.jsonl # Kuratierte Traumberichte
└── README.md
```

---

## ✦ Grundsätze

Dream Oracle folgt klaren ethischen Leitlinien:

- 🕊️ Keine absoluten Wahrheiten — nur Perspektiven und Möglichkeiten
- 🚫 Keine medizinischen oder psychologischen Diagnosen
- 🤝 Die träumende Person steht im Mittelpunkt
- 🛡️ Bei sensiblen Inhalten wird statt einer Deutung ein Hinweis auf professionelle Hilfsangebote angezeigt

---

## ✦ Datenschutz

- **Keine Speicherung** — Traum-Texte werden nicht geloggt, nicht in Datenbanken geschrieben, nicht getrackt
- **Externe API** — Zur Deutung wird der Text an die Hugging Face Inference API gesendet (Llama 3.2). Der API-Anbieter könnte Anfragen temporär protokollieren
- **Gradio Analytics** — Deaktiviert (`analytics_enabled=False`)
- **Zukunftsvision** — Vollständig lokale und offline Deutung, sodass kein Text jemals das Gerät verlässt

---

## ✦ Live Demo

**→ [Dream Oracle auf Hugging Face ausprobieren](https://huggingface.co/spaces/OrangeDev/Narravy-Dream-Oracle)**

---

<div align="center">

*🌙 Built by [Narravy](https://github.com/NarravyBuilds)*

<sub>Ein Projekt von Narravy — wo Technologie auf Träume trifft.</sub>

</div>
