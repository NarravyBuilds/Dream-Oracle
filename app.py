"""Dream Oracle — Hugging Face Spaces Entry Point."""

import json
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
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
SYMBOL_SCORE_THRESHOLD = 0.45


def search_symbols(query: str, top_k: int = 5) -> list[dict]:
    vector = embedder.encode(query).tolist()
    results = db.query_points(
        collection_name="dream_symbols", query=vector, limit=top_k,
        score_threshold=SYMBOL_SCORE_THRESHOLD,
    )
    return [hit.payload for hit in results.points]


def search_reports(query: str, top_k: int = 3) -> list[dict]:
    vector = embedder.encode(query).tolist()
    results = db.query_points(
        collection_name="dream_reports", query=vector, limit=top_k,
        score_threshold=0.25,
    )
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
SYSTEM_INSTRUCTION = (
    "Du bist 'Dream Oracle', ein einfühlsamer Traumdeutungs-Assistent.\n\n"
    "REGELN:\n"
    "- Antworte NUR mit deiner Deutung.\n"
    "- Wiederhole NICHT die Anweisungen, Symbollisten oder den Traum-Text.\n"
    "- Nutze Formulierungen wie 'könnte', 'wird oft assoziiert mit'.\n"
    "- Keine medizinischen oder psychologischen Diagnosen.\n"
    "- Keine Rückfragen stellen.\n\n"
    "PFLICHT-STRUKTUR (genau diese drei Abschnitte, genau in dieser Reihenfolge):\n\n"
    "### Symbolische Bedeutung\n"
    "[Erkläre die Symbole im Traum und ihre möglichen Bedeutungen. "
    "Gehe auf einzelne Elemente ein.]\n\n"
    "### Emotionaler Kontext\n"
    "[In welchen Lebenssituationen treten solche Träume auf? "
    "Welche Gefühle könnten dahinterstehen?]\n\n"
    "### Alternative Lesart\n"
    "[Biete eine andere, unerwartete Perspektive auf den Traum an. "
    "Was könnte er noch bedeuten?]\n\n"
    "FORMATIERUNG:\n"
    "- Verwende EXAKT die drei Überschriften oben mit ### davor.\n"
    "- Hebe nur EINZELNE Schlüsselwörter mit **fett** hervor, z.B. **Transformation**, **Angst**.\n"
    "- Mache NIEMALS ganze Sätze fett. Nur ein einzelnes Wort pro **.\n"
    "- Schreibe in Fließtext-Absätzen, keine Aufzählungslisten.\n"
)


def style_answer(text: str, dream_text: str = "") -> str:
    """Wandelt Markdown in elegante HTML-Formatierung um."""
    import re

    # SCHRITT 1: Traum-Text in Anführungszeichen golden färben (VOR HTML-Konvertierung)
    # Verschiedene Anführungszeichen-Stile abfangen
    text = re.sub(
        r'[„""\"]([^""„"\"]{3,})[""„"\"]',
        r'<gold-quote>\1</gold-quote>',
        text,
    )

    # SCHRITT 2: ### Überschriften → goldene Überschriften
    text = re.sub(
        r"^###\s*(.+)$",
        r'<h3 class="oracle-heading">\1</h3>',
        text,
        flags=re.MULTILINE,
    )

    # SCHRITT 3: **fett** → je nach Länge Überschrift oder Schlüsselwort
    def bold_handler(match):
        content = match.group(1)
        if len(content.split()) >= 4:
            return f'<h3 class="oracle-heading">{content}</h3>'
        return f'<strong class="oracle-keyword">{content}</strong>'

    text = re.sub(r"\*\*(.+?)\*\*", bold_handler, text)

    # SCHRITT 4: Gold-Quote Platzhalter durch echtes HTML ersetzen
    text = text.replace(
        '<gold-quote>',
        '<span class="oracle-keyword" style="font-style:italic;">„'
    )
    text = text.replace('</gold-quote>', '"</span>')

    # SCHRITT 5: Original-Traumtext hervorheben (falls nicht schon in Anführungszeichen)
    if dream_text and len(dream_text) > 5:
        escaped = re.escape(dream_text)
        text = re.sub(
            rf'(?<!„)({escaped})(?!")',
            r'<span class="oracle-keyword" style="font-style:italic;">\1</span>',
            text,
            count=1,
        )

    return text


def call_llm(prompt: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    resp = httpx.post(HF_API_URL, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    raw = _clean_llm_output(raw)
    return raw.strip()


def _clean_llm_output(raw: str) -> str:
    """Entfernt alle Template-Fragmente und Prompt-Leaks aus der LLM-Antwort."""
    import re

    # --- SCHRITT 1: Alles VOR der eigentlichen Deutung abschneiden ---
    # Suche den frühesten sinnvollen Deutungs-Start
    start_markers = [
        "Gesamtdeutung:", "**Gesamtdeutung:**", "### Gesamtdeutung",
        "Deine Aufgabe:", "**Deine Aufgabe:**",
    ]
    for marker in start_markers:
        if marker in raw:
            raw = raw.split(marker, 1)[1]
            break

    # --- SCHRITT 2: Alles NACH unerwünschten Endmarkern abschneiden ---
    # Wenn das Modell am Ende nochmal Prompt-Teile wiederholt
    end_cutoff_patterns = [
        "**Deine Grundsätze:**",
        "Du bist \"Dream Oracle\"",
        "Du bist 'Dream Oracle'",
        "**Passende Symbole aus der Wissensbasis:**",
        "Passende Symbole aus der Wissensbasis",
        "**Ähnliche Traumberichte anderer Menschen:**",
        "Ähnliche Traumberichte anderer Menschen",
    ]
    for pattern in end_cutoff_patterns:
        if pattern in raw:
            raw = raw.split(pattern, 1)[0]

    # --- SCHRITT 3: Einzelne Template-Zeilen entfernen ---
    noise_lines = [
        "### Der Traum",
        "**Der Traum:**",
        "**Der Traum",
        "Der Traum:",
        "### Persönlicher Kontext",
        "**Persönlicher Kontext:**",
        "**Persönlicher Kontext",
        "Persönlicher Kontext:",
        "Kein zusätzlicher Kontext angegeben.",
        "### Deine Aufgabe",
        "**Deine Aufgabe:**",
        "**Deine Aufgabe",
        "Deine Aufgabe:",
        "### Passende Symbole",
        "**Passende Symbole",
        "Passende Symbole aus der Wissensbasis:",
        "### Ähnliche Traumberichte",
        "**Ähnliche Traumberichte",
        "Ähnliche Traumberichte anderer Menschen:",
        "### Rückfrage",
        "**Rückfrage:**",
        "Rückfrage:",
    ]

    cleaned_lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        # Überspringe leere Zeilen am Anfang
        if not cleaned_lines and not stripped:
            continue
        # Überspringe exakte Noise-Matches
        skip = False
        for noise in noise_lines:
            if stripped == noise or stripped == noise.rstrip(":"):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)

    raw = "\n".join(cleaned_lines)

    # --- SCHRITT 4: Rückfrage am Ende komplett entfernen ---
    # Das Modell stellt manchmal am Ende eine Rückfrage
    rq_patterns = [
        r"Rückfrage:.*$",
        r"\*\*Rückfrage:\*\*.*$",
        r"### Rückfrage.*$",
    ]
    for pat in rq_patterns:
        raw = re.sub(pat, "", raw, flags=re.DOTALL)

    return raw


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
    answer = style_answer(answer, dream_text=message)

    motifs = [s.get("symbol", "") for s in symbols if s.get("symbol")]
    badges = motif_badges(motifs)

    return f"{badges}\n\n---\n\n{answer.strip()}"


# --- Gradio UI ---
CSS = """
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
.oracle-heading {
    font-family: 'Cormorant Garamond', serif !important;
    color: #C4A882 !important;
    font-weight: 500 !important;
    font-size: 1.45em !important;
    letter-spacing: 0.5px;
    margin: 24px 0 10px !important;
}
.oracle-keyword {
    color: #C4A882 !important;
    font-weight: 500 !important;
}
/* --- User Bubble --- */
.gradio-container .chatbot .message-wrap .message.user {
    background: linear-gradient(135deg, #2A2520, #1E1B17) !important;
    border: 1px solid rgba(196,168,130,0.3) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 10px 18px !important;
    max-width: 75% !important;
    color: #C4A882 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1em !important;
    font-style: italic;
    letter-spacing: 0.5px;
    line-height: 1.5 !important;
    animation: dream-glow 1.2s ease-out;
}
/* --- Bot Bubble --- */
.gradio-container .chatbot .message-wrap .message.bot {
    background: transparent !important;
    border: none !important;
    padding: 10px 4px !important;
}
/* --- Glow Animation --- */
@keyframes dream-glow {
    0% {
        box-shadow: 0 0 20px rgba(196,168,130,0.5), 0 0 40px rgba(196,168,130,0.2);
        border-color: rgba(196,168,130,0.7);
    }
    50% {
        box-shadow: 0 0 10px rgba(196,168,130,0.3), 0 0 20px rgba(196,168,130,0.1);
    }
    100% {
        box-shadow: none;
        border-color: rgba(196,168,130,0.3);
    }
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
.gradio-container button.primary {
    background: linear-gradient(135deg, #2A2520, #3A3025) !important;
    border: 1px solid #C4A882 !important;
    color: #C4A882 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1em !important;
    font-weight: 500 !important;
    letter-spacing: 1px;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 44px !important;
    width: 100% !important;
}
.gradio-container button.primary:hover {
    background: linear-gradient(135deg, #3A3025, #4A3F30) !important;
    box-shadow: 0 0 12px rgba(196,168,130,0.2) !important;
}
.pdf-export-btn button, button.secondary {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid rgba(196,168,130,0.25) !important;
    color: #7A7168 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 0.95em !important;
    font-weight: 500;
    letter-spacing: 0.5px;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 4px;
}
.pdf-export-btn button:hover, button.secondary:hover {
    border-color: #C4A882 !important;
    color: #C4A882 !important;
    box-shadow: 0 0 8px rgba(196,168,130,0.15);
}
.privacy-notice {
    text-align: center;
    padding: 16px 20px;
    margin-top: 12px;
    font-family: 'Inter', sans-serif;
    font-size: 11.5px;
    line-height: 1.6;
    color: #5A5550;
    letter-spacing: 0.2px;
}
.privacy-notice summary {
    cursor: pointer;
    color: #7A7168;
    font-size: 12px;
    letter-spacing: 0.5px;
    list-style: none;
}
.privacy-notice summary::-webkit-details-marker { display: none; }
.privacy-notice summary::before { content: "🔒 "; }
.privacy-notice .detail-text {
    margin-top: 10px;
    padding: 12px 16px;
    background: rgba(26,23,20,0.5);
    border: 1px solid #2A2520;
    border-radius: 8px;
    text-align: left;
}
"""

with gr.Blocks(css=CSS) as demo:
    gr.HTML("""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500&display=swap">
        <div class="header-wrap">
            <h1>DREAM ORACLE</h1>
            <div class="tagline">Erzähl mir deinen Traum und ich helfe dir, ihn zu verstehen.</div>
            <div class="divider"></div>
        </div>
    """)

    chatbot = gr.Chatbot(height=480, show_label=False, sanitize_html=False)

    msg = gr.Textbox(
        placeholder="Beschreibe deinen Traum...",
        label="Dein Traum",
        lines=2,
    )
    submit_btn = gr.Button(
        "🌙 Deuten",
        variant="primary",
    )
    pdf_btn = gr.Button(
        "📜 Deutung speichern",
        variant="secondary",
        elem_classes=["pdf-export-btn"],
    )
    pdf_file = gr.File(label="Download", visible=False)

    # Speichert die letzte Deutung für den PDF-Export
    last_dream = gr.State("")
    last_answer = gr.State("")

    gr.HTML("""
        <div class="privacy-notice">
            <details>
                <summary>Datenschutzhinweis</summary>
                <div class="detail-text">
                    <strong style="color:#C4A882;">Wie werden deine Daten verarbeitet?</strong><br><br>
                    Dein Traum-Text wird <em>nicht</em> gespeichert. Es gibt keine Datenbank,
                    kein Logging und kein Tracking deiner Eingaben auf diesem Server.<br><br>
                    <strong style="color:#C4A882;">Was passiert bei der Deutung?</strong><br><br>
                    Zur Erzeugung der Deutung wird dein Text an die
                    <strong>Hugging Face Inference API</strong> gesendet, um ein Sprachmodell
                    (Llama 3.2) zu nutzen. Das bedeutet, dass dein Text den Server verlässt
                    und von einem externen Dienst verarbeitet wird. Hugging Face könnte
                    Anfragen temporär protokollieren (z.&nbsp;B. zur Fehlerbehebung oder
                    Missbrauchserkennung).<br><br>
                    <strong style="color:#C4A882;">Empfehlung</strong><br><br>
                    Teile keine persönlichen Informationen (Namen, Orte, Daten), die dich
                    identifizierbar machen. Beschreibe deinen Traum so, dass er für sich
                    allein steht.<br><br>
                    <span style="color:#5A5550;font-size:10.5px;">
                        Dream Oracle befindet sich in aktiver Entwicklung. Unser Ziel ist es,
                        die Deutung zukünftig vollständig lokal und offline durchzuführen,
                        sodass kein Text jemals dieses Gerät verlässt.
                    </span>
                </div>
            </details>
        </div>
    """)

    def respond(message, history, _dream, _ans):
        if not message or not message.strip():
            return "", history or [], _dream, _ans
        history = history or []
        history.append({"role": "user", "content": message})
        answer = interpret(message, history)
        history.append({"role": "assistant", "content": answer})
        return "", history, message, answer

    def export_pdf(dream, answer):
        if not answer:
            return gr.update(visible=False)
        import re
        import os
        import urllib.request
        from datetime import date
        from fpdf import FPDF

        # --- Font Setup ---
        font_dir = "/tmp/dream_oracle_fonts"
        os.makedirs(font_dir, exist_ok=True)
        font_regular = os.path.join(font_dir, "DejaVuSans.ttf")
        font_bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
        font_italic = os.path.join(font_dir, "DejaVuSans-Oblique.ttf")
        # Cormorant Garamond liegt im Repo
        font_title = str(Path(__file__).parent / "app" / "fonts" / "CormorantGaramond-Regular.ttf")

        # DejaVuSans nur herunterladen wenn nicht vorhanden
        dejavu_downloads = [
            (font_regular, "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/src/DejaVuSans.ttf"),
            (font_bold, "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/src/DejaVuSans-Bold.ttf"),
            (font_italic, "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/src/DejaVuSans-Oblique.ttf"),
        ]

        for fpath, url in dejavu_downloads:
            if not os.path.exists(fpath):
                try:
                    urllib.request.urlretrieve(url, fpath)
                except Exception:
                    pass

        # --- Clean HTML for PDF rendering ---
        def clean_for_pdf(html_text):
            """Convert our styled HTML to simple HTML that fpdf2 can render."""
            text = html_text

            # Remove badge spans (the motif tags at the top)
            text = re.sub(r'<span style="background:rgba\(180.*?</span>', '', text)

            # Remove --- dividers
            text = re.sub(r'\n?---\n?', '\n', text)
            text = re.sub(r'^---$', '', text, flags=re.MULTILINE)

            # Convert our oracle-heading h3 to marker
            text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n§H§\1§/H§\n', text, flags=re.DOTALL)

            # Convert oracle-keyword strong to simple <b>
            text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'<b>\1</b>', text)

            # Convert markdown **bold** to <b>
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

            # Convert gold-quote spans to <i>
            text = re.sub(r'<span[^>]*style="[^"]*font-style:italic[^"]*"[^>]*>(.*?)</span>', r'<i>\1</i>', text)

            # Remove all remaining HTML tags except b, i, br
            text = re.sub(r'<(?!/?[bi][ >]|br)[^>]+>', '', text)

            # Clean HTML entities
            text = text.replace('&middot;', '-').replace('&nbsp;', ' ')

            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)

            return text.strip()

        cleaned = clean_for_pdf(answer)

        # Split into heading blocks and text blocks
        parts = re.split(r'(§H§.*?§/H§)', cleaned, flags=re.DOTALL)

        # --- Build PDF ---
        class DarkPDF(FPDF):
            def header(self):
                self.set_fill_color(26, 23, 20)
                self.rect(0, 0, 210, 297, 'F')

        pdf = DarkPDF()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.add_page()

        # Register Unicode fonts if available
        use_unicode = os.path.exists(font_regular)
        if use_unicode:
            pdf.add_font("Dream", "", font_regular)
            pdf.add_font("Dream", "B", font_bold)
            pdf.add_font("Dream", "I", font_italic)
            fn = "Dream"
        else:
            fn = "Helvetica"

        # Register Cormorant Garamond for title
        use_title_font = os.path.exists(font_title)
        if use_title_font:
            pdf.add_font("Cormorant", "", font_title)
            tfn = "Cormorant"
        else:
            tfn = fn

        # --- HEADER: Cormorant Garamond like the website ---
        pdf.set_text_color(196, 168, 130)
        pdf.set_font(tfn, "", 36)
        pdf.cell(0, 26, "DREAM  ORACLE", align="C", new_x="LMARGIN", new_y="NEXT")

        # Tagline
        pdf.set_text_color(122, 113, 104)
        pdf.set_font(fn, "", 9)
        tagline = "Erzahl mir deinen Traum und ich helfe dir, ihn zu verstehen."
        if use_unicode:
            tagline = "Erzähl mir deinen Traum und ich helfe dir, ihn zu verstehen."
        pdf.cell(0, 6, tagline, align="C", new_x="LMARGIN", new_y="NEXT")

        # Gold divider
        pdf.ln(6)
        pdf.set_draw_color(196, 168, 130)
        pdf.line(75, pdf.get_y(), 135, pdf.get_y())
        pdf.ln(12)

        # --- DREAM INPUT ---
        pdf.set_text_color(196, 168, 130)
        pdf.set_font(fn, "I", 14)
        dream_text = dream if use_unicode else dream.encode('latin-1', 'replace').decode('latin-1')
        x_start = 25
        box_width = 160
        pdf.set_x(x_start)
        y_before = pdf.get_y()
        pdf.multi_cell(box_width, 8, f'"{dream_text}"', align="C")
        y_after = pdf.get_y()
        pdf.set_draw_color(196, 168, 130)
        pdf.set_line_width(0.3)
        pdf.rect(x_start - 5, y_before - 3, box_width + 10, (y_after - y_before) + 6)
        pdf.ln(12)

        # --- CONTENT ---
        left_margin = 15
        content_width = 180

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if pdf.get_y() > 265:
                pdf.add_page()

            # Heading
            h_match = re.match(r'§H§(.*?)§/H§', part, re.DOTALL)
            if h_match:
                heading = h_match.group(1).strip().rstrip(":")
                heading = re.sub(r'<[^>]+>', '', heading)  # strip any tags inside heading
                if not use_unicode:
                    heading = heading.encode('latin-1', 'replace').decode('latin-1')
                pdf.ln(5)
                pdf.set_x(left_margin)
                pdf.set_text_color(196, 168, 130)
                pdf.set_font(fn, "B", 14)
                pdf.multi_cell(content_width, 8, heading)
                pdf.set_draw_color(196, 168, 130)
                pdf.line(left_margin, pdf.get_y() + 1, left_margin + 40, pdf.get_y() + 1)
                pdf.ln(5)
            else:
                # Regular text with inline bold/italic via write_html
                # Clean for write_html: only <b>, <i>, <br> allowed
                block = part.strip()
                block = re.sub(r'<(?!/?[bi][ >]|br)[^>]+>', '', block)
                # Remove bullet markers
                block = re.sub(r'^\s*[\-\*]\s*', '', block, flags=re.MULTILINE)
                # Convert newlines to <br> for write_html
                block = block.replace('\n\n', '<br><br>').replace('\n', '<br>')
                block = re.sub(r'(<br>){3,}', '<br><br>', block)

                if not use_unicode:
                    block = block.encode('latin-1', 'replace').decode('latin-1')

                if block.strip() and block.strip() != '<br>':
                    pdf.set_x(left_margin)
                    pdf.set_text_color(212, 204, 196)
                    pdf.set_font(fn, "", 10.5)
                    # write_html handles <b> and <i> inline
                    pdf.write_html(f'<font color="#D4CCC4">{block}</font>')
                    pdf.ln(4)

        # --- FOOTER ---
        pdf.ln(8)
        if pdf.get_y() > 275:
            pdf.add_page()
        pdf.set_draw_color(42, 37, 32)
        pdf.line(left_margin, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)
        pdf.set_text_color(90, 85, 80)
        pdf.set_font(fn, "", 8)
        footer_date = date.today().strftime('%d.%m.%Y')
        pdf.cell(0, 5, f"Dream Oracle  |  {footer_date}", align="C")

        path = "/tmp/dream_oracle.pdf"
        pdf.output(path)
        return gr.update(value=path, visible=True)

    msg.submit(respond, [msg, chatbot, last_dream, last_answer], [msg, chatbot, last_dream, last_answer])
    submit_btn.click(respond, [msg, chatbot, last_dream, last_answer], [msg, chatbot, last_dream, last_answer])
    pdf_btn.click(export_pdf, [last_dream, last_answer], [pdf_file])

if __name__ == "__main__":
    demo.launch()
