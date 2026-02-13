from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recipe_ingestion import (
    EMBED_DIM,
    ensure_data_dir,
    generate_rag_response_with_gemini,
    ingest_recipe_from_text,
    ingest_recipe_from_youtube,
    load_embedding_model,
    load_index,
    load_metadata,
    search_similar_recipes,
)

app = FastAPI(title="Little Chef API", version="0.1.0")
FRONTEND_DIR = Path(__file__).parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class IngestYoutubeRequest(BaseModel):
    link: str = Field(..., min_length=1)
    cookies_path: str = "youtube_cookies.txt"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 3


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 3
    api_key: Optional[str] = None


class RecipeResult(BaseModel):
    title: str
    text: str
    steps: List[str]
    num_steps: int
    distance: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[RecipeResult]


class MessageResponse(BaseModel):
    message: str


class RagResponse(BaseModel):
    response: str


class StatsResponse(BaseModel):
    recipes: int


@lru_cache(maxsize=1)
def get_state():
    ensure_data_dir()
    model = load_embedding_model()
    index = load_index(EMBED_DIM)
    metadata = load_metadata()
    return model, index, metadata


@app.get("/health", response_model=MessageResponse)
def health() -> MessageResponse:
    return MessageResponse(message="ok")


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    _, index, metadata = get_state()
    return StatsResponse(recipes=len(metadata))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/ingest/text", response_model=MessageResponse)
def ingest_text(req: IngestTextRequest) -> MessageResponse:
    model, index, metadata = get_state()
    ingest_recipe_from_text(index, metadata, model, req.text.strip())
    return MessageResponse(message="recipe stored")


@app.post("/ingest/youtube", response_model=MessageResponse)
def ingest_youtube(req: IngestYoutubeRequest) -> MessageResponse:
    model, index, metadata = get_state()
    try:
        ingest_recipe_from_youtube(
            index,
            metadata,
            model,
            req.link.strip(),
            cookies_path=req.cookies_path,
            use_whisper_fallback=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MessageResponse(message="recipe stored")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    model, index, metadata = get_state()
    results = search_similar_recipes(index, metadata, model, req.query.strip(), req.top_k)
    return SearchResponse(results=[RecipeResult(**r) for r in results])


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest) -> RagResponse:
    model, index, metadata = get_state()
    try:
        response_text, _ = generate_rag_response_with_gemini(
            index,
            metadata,
            model,
            req.query.strip(),
            top_k=req.top_k,
            api_key=req.api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RagResponse(response=response_text)
