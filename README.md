# Little-Chef
A Little-Chef who will keep track of all your recipes from any platform and help you navigate and use them to make amazing food!

## Simple Recipe App (FAISS)
This repo includes a tiny app to ingest recipes and search them with a FAISS vector store.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### How it works
- Paste a recipe in the **Ingest** tab to store it.
- Use the **Search** tab to find the closest recipe based on your query.
- (Optional) Set `GOOGLE_API_KEY` to enable the Gemini RAG button.
- (Optional) YouTube ingestion uses `openai-whisper` + `yt-dlp` and requires `ffmpeg` installed locally.

### Notebook code reuse
Core ingestion + FAISS + Gemini RAG logic is extracted into `recipe_ingestion.py` and used by the app.
