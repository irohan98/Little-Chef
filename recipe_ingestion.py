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
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except Exception:  # pragma: no cover
    YouTubeTranscriptApi = None
    TranscriptsDisabled = None
    NoTranscriptFound = None

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


# ----- YouTube scraping (v2 from notebook) -----

def is_youtube_link(link: str) -> bool:
    return "youtube.com" in link or "youtu.be" in link


def get_video_id(link: str) -> str:
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(link)
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return ""


def try_youtube_captions(video_id: str) -> str:
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed.")
    try:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except AttributeError:
            transcript = YouTubeTranscriptApi().get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception:
        return ""


def load_whisper_model():
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed.")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return whisper.load_model("base", device="cpu")


def transcribe_audio_with_yt_dlp(
    link: str,
    use_cookies: bool = False,
    cookies_path: str = "youtube_cookies.txt",
) -> str:
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed.")
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed.")

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": False,
        "outtmpl": "/tmp/audio.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    if use_cookies:
        ydl_opts["cookiefile"] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        import glob

        audio_files = glob.glob("/tmp/audio*.mp3")
        if not audio_files:
            raise FileNotFoundError("No audio file found.")
        audio_path = audio_files[0]

        model = load_whisper_model()
        result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "")
    except Exception:
        return ""


def ingest_recipe_smart(
    link: str,
    cookies_path: str = "youtube_cookies.txt",
    use_whisper_fallback: bool = True,
) -> str:
    video_id = get_video_id(link)
    if not video_id:
        print("❌ Invalid YouTube link.")
        return ""

    print("🔍 Trying captions...")
    text = try_youtube_captions(video_id)
    if text:
        print("✅ Captions found.")
        return text

    env_disable = os.environ.get("LC_DISABLE_WHISPER", "").strip().lower() in {"1", "true", "yes"}
    if not use_whisper_fallback or env_disable:
        print("⚠️ Captions unavailable. Whisper fallback is disabled.")
        return ""

    print("🌀 Trying audio transcription without cookies...")
    text = transcribe_audio_with_yt_dlp(link, use_cookies=False)
    if text:
        print("✅ Transcription without cookies succeeded.")
        return text

    print("🔐 Trying with cookies...")
    text = transcribe_audio_with_yt_dlp(link, use_cookies=True, cookies_path=cookies_path)
    if text:
        print("✅ Transcription with cookies succeeded.")
        return text

    print("❌ All attempts failed.")
    return ""


def ingest_recipe_from_youtube(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    link: str,
    cookies_path: str = "youtube_cookies.txt",
    use_whisper_fallback: bool = True,
) -> Dict[str, Any]:
    if not is_youtube_link(link):
        raise ValueError("Invalid YouTube URL.")

    text = ingest_recipe_smart(
        link,
        cookies_path=cookies_path,
        use_whisper_fallback=use_whisper_fallback,
    )

    if not text:
        raise RuntimeError("Failed to get captions or transcribe video.")

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

    model_g = genai.GenerativeModel("gemini-2.5-flash")
    response = model_g.generate_content(prompt)
    return response.text, results
