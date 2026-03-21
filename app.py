"""Dream Oracle — Hugging Face Spaces Entry Point."""

import json
import os
from pathlib import Path

import gradio as gr
import httpx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- Konfiguration ---
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
VECTOR_SIZE = 384
DATA_DIR = Path(__file__).parent / "data"
PROMPT_PATH = Path(__file__).parent / "app" / "prompts" / "interpret_prompt.txt"

# --- Modelle & DB laden ---
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = QdrantClient(path="./qdrant_data")

BLOCKED_PATTERNS = ["suizid", "selbstmord", "umbringen", "selbstverletzung"]
SAFETY_MSG = (
    "Dein Text enthält sensible Inhalte. "
    "Wenn du dich in einer Krise befindest, wende dich bitte an die Telefonseelsorge "
    "(0800 111 0 111 / 0800 111 0 222) oder an eine Fachperson deines Vertrauens."
)


# --- Daten-Ingestion beim Start ---
def ingest():
    """Baut den Suchindex beim Start auf."""
    existing = [c.name for c in db.get_collections().collections]

    for name in ["dream_symbols", "dream_reports"]:
        if name not in existing:
            db.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    # Symbole
    if db.count("dream_symbols").count == 0:
        symbols = json.loads((DATA_DIR / "symbols.json").read_text(encoding="utf-8"))
        points = []
        for i, sym in enumerate(symbols):
            text = f"{sym['symbol']}: {', '.join(sym['mögliche_bedeutungen'])}"
            vector = embedder.encode(text).tolist()
            points.append(PointStruct(id=i + 1, vector=vector, payload=sym))
        db.upsert(collection_name="dream_symbols", points=points)
        print(f"{len(points)} Symbole geladen.")

    # Berichte
    if db.count("dream_reports").count == 0:
        lines = (DATA_DIR / "dream_reports.jsonl").read_text(encoding="utf-8").strip().split("\n")
        points = []
        for line in lines:
            r = json.loads(line)
            vector = embedder.encode(r["text"]).tolist()
            points.append(PointStruct(id=r["id"], vector=vector, payload={
                "text": r["text"], "emotion": r["emotion"],
                "motive": r["motive"], "subjektive_deutung": r.get("subjektive_deutung", ""),
            }))
        db.upsert(collection_name="dream_reports", points=points)
        print(f"{len(points)} Berichte geladen.")


ingest()


# --- Such-Funktionen ---
def search_symbols(query: str, top_k: int = 5) -> list[dict]:
    vector = embedder.encode(query).tolist()
    results = db.query_points(collection_name="dream_symbols", query=vector, limit=top_k)
    return [hit.payload for hit in results.points]


def search_reports(query: str, top_k: int = 3) -> list[dict]:
    vector = embedder.encode(query).tolist()
    results = db.query_points(collection_name="dream_reports", query=vector, limit=top_k)
    return [hit.payload for hit in results.points]


# --- Prompt bauen ---
def build_prompt(dream_text: str, symbols: list[dict], reports: list[dict]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")

    symbols_text = "\n".join(
        f"- {s.get('symbol', '?')}: {', '.join(s.get('mögliche_bedeutungen', []))}"
        for s in symbols
    )
    reports_text = "\n".join(
        f"- \"{r.get('text', '')[:120]}...\" (Emotion: {r.get('emotion', '?')})"
        for r in reports
    )

    return template.format(
        dream_text=dream_text,
        context="Kein zusätzlicher Kontext angegeben.",
        symbols=symbols_text or "Keine passenden Symbole gefunden.",
        reports=reports_text or "Keine ähnlichen Berichte gefunden.",
    )


# --- LLM aufrufen ---
def call_llm(prompt: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    resp = httpx.post(HF_API_URL, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --- UI-Hilfsfunktionen ---
def motif_badges(motifs: list[str]) -> str:
    badges = []
    for m in motifs:
        badges.append(
            f'<span style="background:rgba(180,160,130,0.15);color:#C4A882;'
            f'padding:5px 16px;border-radius:24px;font-size:13px;'
            f'font-weight:500;display:inline-block;margin:3px 4px;'
            f'border:1px solid rgba(180,160,130,0.3);'
            f'letter-spacing:0.5px;">{m}</span>'
        )
    return " ".join(badges)


def interpret(message: str, history: list[dict]) -> str:
    # Safety Check
    lower = message.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower:
            return SAFETY_MSG

    symbols = search_symbols(message, top_k=5)
    reports = search_reports(message, top_k=3)
    prompt = build_prompt(message, symbols, reports)
    answer = call_llm(prompt)

    motifs = [s.get("symbol", "") for s in symbols if s.get("symbol")]
    badges = motif_badges(motifs)

    return f"{badges}\n\n---\n\n{answer.strip()}"


# --- Gradio UI ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500&display=swap');

.gradio-container {
    max-width: 780px !important;
    margin: auto !important;
    background: #1A1714 !important;
    font-family: 'Inter', sans-serif !important;
}
.header-wrap {
    text-align: center;
    padding: 40px 20px 10px;
}
.header-wrap h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    font-size: 3em !important;
    color: #C4A882 !important;
    letter-spacing: 3px;
    margin: 0 !important;
}
.header-wrap .tagline {
    font-family: 'Inter', sans-serif;
    color: #7A7168;
    font-size: 0.9em;
    font-weight: 300;
    letter-spacing: 1px;
    margin-top: 6px;
}
.header-wrap .divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C4A882, transparent);
    margin: 18px auto 0;
}
.gradio-container .chatbot {
    background: #1A1714 !important;
    border: 1px solid #2A2520 !important;
    border-radius: 12px !important;
}
.message {
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.75 !important;
    color: #D4CCC4 !important;
    letter-spacing: 0.2px;
}
.message h3, .message strong {
    font-family: 'Cormorant Garamond', serif !important;
    color: #C4A882 !important;
    font-weight: 500 !important;
    font-size: 1.15em !important;
    letter-spacing: 0.5px;
}
.message hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #3A3530, transparent) !important;
    margin: 20px 0 !important;
}
.gradio-container textarea {
    background: #211E1A !important;
    border: 1px solid #2A2520 !important;
    border-radius: 10px !important;
    color: #D4CCC4 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 14px !important;
}
.gradio-container textarea:focus {
    border-color: #C4A882 !important;
    box-shadow: 0 0 0 1px rgba(196,168,130,0.15) !important;
}
.gradio-container label {
    font-family: 'Cormorant Garamond', serif !important;
    color: #8A8078 !important;
    font-size: 1.05em !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1A1714; }
::-webkit-scrollbar-thumb { background: #3A3530; border-radius: 3px; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS) as demo:
    gr.HTML("""
        <div class="header-wrap">
            <h1>DREAM ORACLE</h1>
            <div class="tagline">Erzähl mir deinen Traum und ich helfe dir, ihn zu verstehen.</div>
            <div class="divider"></div>
        </div>
    """)

    chatbot = gr.Chatbot(height=480, show_label=False)
    msg = gr.Textbox(
        placeholder="Beschreibe deinen Traum...",
        label="Dein Traum",
        lines=2,
    )

    def respond(message, history):
        history = history or []
        history.append({"role": "user", "content": message})
        answer = interpret(message, history)
        history.append({"role": "assistant", "content": answer})
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
