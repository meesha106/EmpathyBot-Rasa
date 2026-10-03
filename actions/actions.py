# actions/actions.py
import os
import json
import numpy as np
import pdfplumber
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Directories & files
KB_DIR = "retrieval"
PDF_DIR = "data/pdfs"
EMBED_FILE = os.path.join(KB_DIR, "kb_embeddings.npz")
KB_META = os.path.join(KB_DIR, "kb_meta.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality

# Initialize summarizer globally
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def ensure_dirs():
    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def load_pdfs_text(pdf_dir: str) -> List[Dict[str, str]]:
    """Extract text paragraphs from PDFs."""
    texts = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = (p.extract_text() or "").strip()
                if txt:
                    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
                    for part in parts:
                        if len(part) > 40:  # skip tiny fragments
                            texts.append({"source": fname, "text": part})
    return texts

def build_or_load_kb(model: SentenceTransformer):
    """Load embeddings and metadata or build them from PDFs."""
    ensure_dirs()
    if os.path.exists(EMBED_FILE) and os.path.exists(KB_META):
        arr = np.load(EMBED_FILE)
        embeddings = arr["embeddings"]
        with open(KB_META, "r", encoding="utf8") as f:
            meta = json.load(f)
        return embeddings, meta

    meta_texts = load_pdfs_text(PDF_DIR)
    if not meta_texts:
        raise RuntimeError(f"No PDF texts found in {PDF_DIR}. Place your PDFs there.")

    texts = [m["text"] for m in meta_texts]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.savez_compressed(EMBED_FILE, embeddings=embeddings)
    with open(KB_META, "w", encoding="utf8") as f:
        json.dump(meta_texts, f, ensure_ascii=False, indent=2)
    return embeddings, meta_texts

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Summarizes long text into a concise 2–3 sentence version.
    max_length controls the approximate output length (tokens).
    """
    if len(text.split()) < 50:  # short text, no need to summarize
        return text
    try:
        summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print("Summarization failed:", e)
        return text  # fallback to original

class ActionRetrieveNutrition(Action):
    def name(self) -> Text:
        return "action_retrieve_nutrition"

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        try:
            self.embeddings, self.meta = build_or_load_kb(self.model)
        except Exception as e:
            print("KB build/load failed:", e)
            self.embeddings, self.meta = None, []

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        last_user = tracker.latest_message.get("text", "")
        if not last_user:
            dispatcher.utter_message(text="Can you tell me your question?")
            return []

        empathic_starts = [
            "I hear you — that can feel worrying. Here's info that may help:",
            "I understand — here's what the guidelines say:",
            "I get your concern. According to the guidelines:"
        ]

        try:
            if self.embeddings is None or len(self.meta) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't load the guideline files. Make sure PDFs are in data/pdfs.")
                return []

            # Encode query and compute similarity
            q_emb = self.model.encode(last_user, convert_to_numpy=True)
            scores = util.cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            best_item = self.meta[best_idx]
            paragraph = best_item.get("text", "")
            source = best_item.get("source", "uploaded guideline")

            # Fallback if score is low
            if best_score < 0.25:
                fallback_msg = (
                    "I might not have an exact match, but generally:\n"
                    "- Spread carbohydrates throughout the day (3 meals + 2–3 snacks).\n"
                    "- Prefer complex carbs and fiber-rich foods.\n"
                    "- Include moderate protein and healthy fats.\n"
                    "This is based on national GDM guidelines."
                )
                dispatcher.utter_message(text=f"{empathic_starts[0]}\n\n{fallback_msg}")
                return []

            # Summarize long paragraphs for readability
            paragraph = summarize_text(paragraph)

            reply = f"{empathic_starts[1]}\n\n{paragraph}\n\n(Source: {source})"
            dispatcher.utter_message(text=reply)
            return []

        except Exception as e:
            dispatcher.utter_message(text="Sorry, something went wrong. Try rephrasing your question.")
            print("Action error:", e)
            return []

class ActionGenerateEmpathy(Action):
    def name(self) -> str:
        return "action_generate_empathy"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        last_user = tracker.latest_message.get("text", "")
        if not last_user:
            dispatcher.utter_message(text="I hear you. Can you tell me more?")
        else:
            dispatcher.utter_message(
                text="I understand — that must be tough. Remember to breathe and take care of yourself."
            )
        return []
