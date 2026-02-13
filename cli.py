import argparse
import sys

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


def cmd_ingest_text(args: argparse.Namespace) -> None:
    ensure_data_dir()
    model = load_embedding_model()
    index = load_index(EMBED_DIM)
    metadata = load_metadata()

    text = args.text
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print("No recipe text provided.")
        return

    recipe = ingest_recipe_from_text(index, metadata, model, text)
    print(f"Stored: {recipe.get('title', 'Untitled')} (steps: {recipe.get('num_steps', 0)})")


def cmd_ingest_youtube(args: argparse.Namespace) -> None:
    ensure_data_dir()
    model = load_embedding_model()
    index = load_index(EMBED_DIM)
    metadata = load_metadata()

    recipe = ingest_recipe_from_youtube(
        index,
        metadata,
        model,
        args.link,
        cookies_path=args.cookies,
        use_whisper_fallback=True,
    )
    print(f"Stored from YouTube: {recipe.get('title', 'Untitled')}")


def cmd_search(args: argparse.Namespace) -> None:
    ensure_data_dir()
    model = load_embedding_model()
    index = load_index(EMBED_DIM)
    metadata = load_metadata()

    results = search_similar_recipes(index, metadata, model, args.query, top_k=args.top_k)
    if not results:
        print("No recipes found.")
        return

    for i, item in enumerate(results, start=1):
        print(f"\nMatch {i}: {item['title']}")
        print(f"Distance: {item['distance']:.4f}")
        print(item["text"])


def cmd_rag(args: argparse.Namespace) -> None:
    ensure_data_dir()
    model = load_embedding_model()
    index = load_index(EMBED_DIM)
    metadata = load_metadata()

    response_text, _ = generate_rag_response_with_gemini(
        index,
        metadata,
        model,
        args.query,
        top_k=args.top_k,
        api_key=args.api_key,
    )
    print(response_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Little-Chef CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest-text", help="Ingest recipe text")
    p_ingest.add_argument("--text", help="Recipe text. If omitted, reads stdin.")
    p_ingest.set_defaults(func=cmd_ingest_text)

    p_yt = sub.add_parser("ingest-youtube", help="Ingest recipe from YouTube link")
    p_yt.add_argument("link", help="YouTube link")
    p_yt.add_argument("--cookies", default="youtube_cookies.txt", help="Cookies file path")
    p_yt.set_defaults(func=cmd_ingest_youtube)

    p_search = sub.add_parser("search", help="Search recipes")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=3, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    p_rag = sub.add_parser("rag", help="Gemini RAG search")
    p_rag.add_argument("query", help="User query")
    p_rag.add_argument("--top-k", type=int, default=3, help="Number of results")
    p_rag.add_argument("--api-key", default=None, help="Gemini API key (optional)")
    p_rag.set_defaults(func=cmd_rag)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
