import streamlit as st
from recipe_ingestion import (
    EMBED_DIM,
    ensure_data_dir,
    generate_rag_response_with_gemini,
    ingest_recipe_from_text,
    ingest_recipe_from_youtube,
    load_embedding_model,
    load_index,
    load_metadata,
    rebuild_index,
    search_similar_recipes,
    save_metadata,
)


def main() -> None:
    st.set_page_config(page_title="Little Chef", page_icon="🍳", layout="centered")
    st.title("Little Chef")
    st.write("A tiny recipe ingestion + search app using FAISS.")

    ensure_data_dir()
    model = load_embedding_model()
    metadata = load_metadata()
    index = load_index(EMBED_DIM)

    st.sidebar.header("Storage")
    st.sidebar.write(f"Recipes stored: {len(metadata)}")
    st.sidebar.write(f"Index vectors: {index.ntotal}")

    if index.ntotal != len(metadata):
        st.sidebar.warning("Index and metadata count do not match.")
        if st.sidebar.button("Rebuild Index"):
            index = rebuild_index(metadata, model)
            save_metadata(metadata)
            st.sidebar.success("Index rebuilt.")

    tab_ingest, tab_search = st.tabs(["Ingest", "Search"])

    with tab_ingest:
        st.subheader("Add a recipe")
        recipe_text = st.text_area("Paste a recipe", height=240, placeholder="Title\n1. Step one...\n2. Step two...")
        if st.button("Save Recipe"):
            if not recipe_text.strip():
                st.error("Please paste a recipe.")
            else:
                ingest_recipe_from_text(index, metadata, model, recipe_text.strip())
                st.success("Recipe saved.")

        st.markdown("---")
        st.subheader("Add from YouTube (v1)")
        st.write("Requires ffmpeg, `openai-whisper`, and `yt-dlp` installed.")
        yt_link = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
        cookies_path = st.text_input("Cookies file (optional)", value="youtube_cookies.txt")
        if st.button("Ingest YouTube Recipe"):
            if not yt_link.strip():
                st.error("Enter a YouTube link.")
            else:
                try:
                    ingest_recipe_from_youtube(index, metadata, model, yt_link.strip(), cookies_path=cookies_path)
                except Exception as e:
                    st.error(str(e))
                else:
                    st.success("YouTube recipe ingested.")

    with tab_search:
        st.subheader("Find a recipe")
        query = st.text_input("What do you want to make?", placeholder="e.g., eggs under 30 minutes")
        top_k = st.slider("Top K", min_value=1, max_value=5, value=3)
        if st.button("Search"):
            if not query.strip():
                st.error("Enter a search query.")
            else:
                results = search_similar_recipes(index, metadata, model, query.strip(), top_k)
                if not results:
                    st.info("No recipes found. Add some recipes first.")
                else:
                    for i, item in enumerate(results, start=1):
                        st.markdown(f"### Match {i}: {item['title']}")
                        st.write(f"Distance: {item['distance']:.4f}")
                        st.write(item["text"])

        st.markdown("---")
        st.subheader("Gemini RAG (optional)")
        st.write("Requires `GOOGLE_API_KEY` in your environment.")
        if st.button("Ask Gemini"):
            if not query.strip():
                st.error("Enter a search query first.")
            else:
                try:
                    response_text, _ = generate_rag_response_with_gemini(
                        index, metadata, model, query.strip(), top_k=top_k
                    )
                except Exception as e:
                    st.error(str(e))
                else:
                    st.markdown("### Gemini Response")
                    st.write(response_text)


if __name__ == "__main__":
    main()
