import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None

try:
    import yt_dlp
except Exception:  # pragma: no cover
    yt_dlp = None

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "recipes.faiss"
META_PATH = DATA_DIR / "recipes.jsonl"


# ----- Core parsing helpers (from notebook logic) -----

def extract_recipe_from_text(text: str) -> Dict[str, Any]:
    steps = re.split(r"\d+\.", text)
    steps = [step.strip() for step in steps if len(step.strip()) > 10]
    return {
        "raw_text": text,
        "steps": steps,
        "num_steps": len(steps),
        "tags": [],
    }


def extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+[\).\-]?\s+", line):
            continue
        return line[:80]
    return "Untitled Recipe"


# ----- YouTube scraping (v1 from notebook) -----

def is_youtube_link(link: str) -> bool:
    return "youtube.com" in link or "youtu.be" in link


def load_whisper_model():
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed.")
    return whisper.load_model("base")


def transcribe_youtube(link: str, cookies_path: str = "youtube_cookies.txt") -> str:
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed.")
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed.")

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": False,
        "outtmpl": "/tmp/audio.%(ext)s",
        "cookiefile": cookies_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        audio_path = None
        for fname in os.listdir("/tmp"):
            if fname.startswith("audio") and fname.endswith(".mp3"):
                audio_path = os.path.join("/tmp", fname)
                break
        if not audio_path:
            raise FileNotFoundError("Audio file not found after download.")

        model = load_whisper_model()
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception:
        return ""


def ingest_recipe_from_youtube(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    link: str,
    cookies_path: str = "youtube_cookies.txt",
) -> Dict[str, Any]:
    if not is_youtube_link(link):
        raise ValueError("Invalid YouTube URL.")

    text = transcribe_youtube(link, cookies_path)
    if not text:
        raise RuntimeError("Failed to transcribe video.")

    return ingest_recipe_from_text(index, metadata, model, text)


# ----- Embeddings + storage -----

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    emb = model.encode([text])
    return np.array(emb, dtype="float32")


def load_metadata() -> List[Dict[str, Any]]:
    if not META_PATH.exists():
        return []
    items: List[Dict[str, Any]] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_metadata(items: List[Dict[str, Any]]) -> None:
    with META_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def load_index(dim: int = EMBED_DIM) -> faiss.Index:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatL2(dim)


def save_index(index: faiss.Index) -> None:
    faiss.write_index(index, str(INDEX_PATH))


def rebuild_index(metadata: List[Dict[str, Any]], model: SentenceTransformer) -> faiss.Index:
    index = faiss.IndexFlatL2(EMBED_DIM)
    if not metadata:
        save_index(index)
        return index

    embeddings = [embed_text(model, item["text"]) for item in metadata]
    matrix = np.vstack(embeddings)
    index.add(matrix)
    save_index(index)
    return index


def store_recipe_to_vector_db(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    recipe: Dict[str, Any],
) -> None:
    embedding = embed_text(model, recipe["raw_text"])

    index.add(embedding)
    metadata.append({
        "title": recipe.get("title") or extract_title(recipe["raw_text"]),
        "text": recipe["raw_text"],
        "steps": recipe["steps"],
        "num_steps": recipe["num_steps"],
        "tags": recipe.get("tags", []),
    })

    save_index(index)
    save_metadata(metadata)


def ingest_recipe_from_text(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    raw_text: str,
) -> Dict[str, Any]:
    recipe = extract_recipe_from_text(raw_text)
    recipe["raw_text"] = raw_text
    recipe["title"] = extract_title(raw_text)
    store_recipe_to_vector_db(index, metadata, model, recipe)
    return recipe


# ----- Search -----

def search_similar_recipes(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if index.ntotal == 0:
        return []
    query_embedding = embed_text(model, query)
    distances, indices = index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["distance"] = float(dist)
        results.append(item)
    return results


# ----- Gemini RAG (from notebook idea) -----

def generate_rag_response_with_gemini(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    user_query: str,
    top_k: int = 3,
    api_key: str | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY.")

    genai.configure(api_key=key)

    results = search_similar_recipes(index, metadata, model, user_query, top_k=top_k)
    context_snippets = []
    for recipe in results:
        snippet = "Steps:\n" + "\n".join(recipe["steps"][:6])
        context_snippets.append(snippet)

    context = "\n\n---\n\n".join(context_snippets)
    prompt = (
        "You are a helpful recipe assistant.\n\n"
        f"User request: {user_query}\n\n"
        "Here are some recipes:\n\n"
        f"{context}\n\n"
        "Please recommend the best-fitting recipe and explain why it was chosen."
    )

    model_g = genai.GenerativeModel("gemini-pro")
    response = model_g.generate_content(prompt)
    return response.text, results
