# Little-Chef
A Little-Chef who will keep track of all your recipes from any platform and help you navigate and use them to make amazing food!

## Simple Recipe App (FAISS)
This repo includes a tiny app to ingest recipes and search them with a FAISS vector store.

### FastAPI
```bash
export OMP_NUM_THREADS=1     
export MKL_NUM_THREADS=1
export UVLOOP_NO_EXTENSIONS=1
uvicorn api:app --reload --loop asyncio
```

Example requests:
```bash
curl -X POST http://127.0.0.1:8000/ingest/text \\
  -H \"Content-Type: application/json\" \\
  -d '{\"text\": \"Title\\n1. Step one...\\n2. Step two...\"}'

curl -X POST http://127.0.0.1:8000/search \\
  -H \"Content-Type: application/json\" \\
  -d '{\"query\": \"salmon under 30 minutes\", \"top_k\": 3}'
```

Open the UI at `http://127.0.0.1:8000/`.

### CLI (no Streamlit)
```bash
python cli.py ingest-text --text "Title\n1. Step one...\n2. Step two..."
python cli.py search "salmon under 30 minutes" --top-k 3
python cli.py ingest-youtube "https://www.youtube.com/watch?v=..."
```

### How it works
- Paste a recipe in the **Ingest** tab to store it.
- Set `GOOGLE_API_KEY` to enable the Gemini RAG to retrieve the recipe closest to your prompt from the stored recipies.

### Notebook code reuse
Core ingestion + FAISS + Gemini RAG logic is extracted into `recipe_ingestion.py` and used by the app.
