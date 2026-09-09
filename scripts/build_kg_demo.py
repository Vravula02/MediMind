"""Utility script to build and visualize the knowledge graph from cached chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from medical_graph_rag.knowledge_graph.graph_viz import GraphVisualizer
from medical_graph_rag.knowledge_graph.knowledge_graph import KnowledgeGraph

DEFAULT_CHUNK_FILE = Path("data/output/processed_pmc_data/pmc_chunks.json")


def load_documents(chunk_file: Path, limit: int | None = 200) -> list[Document]:
    """Load preprocessed chunk documents and wrap them as LangChain Documents."""
    with chunk_file.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    documents_section = payload.get("documents", [])
    documents = []
    for entry in documents_section:
        documents.append(
            Document(
                page_content=entry.get("content", ""),
                metadata=entry.get("metadata", {}),
            )
        )
        if limit and len(documents) >= limit:
            break

    if not documents:
        raise ValueError(f"No documents loaded from {chunk_file}")

    return documents


def build_and_visualize_kg(
    chunk_file: Path, limit: int | None, output_path: Path | None
) -> None:
    """Construct the knowledge graph and optionally save a visualization."""
    docs = load_documents(chunk_file, limit)

    kg = KnowledgeGraph(cache_dir="demo_cache")
    graph = kg.build_knowledge_graph(docs)
    stats = kg.get_stats()

    print("Knowledge Graph Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if output_path:
        visualizer = GraphVisualizer()
        traversal = list(range(min(len(docs), max(1, graph.number_of_nodes()))))
        image_buffer = visualizer.visualize_traversal(graph, traversal)
        if image_buffer is None:
            print("Visualization not generated; traversal path may be empty.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            fh.write(image_buffer.read())
        print(f"Saved traversal visualization to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and visualize the knowledge graph from cached chunks."
    )
    parser.add_argument(
        "--chunk-file",
        type=Path,
        default=DEFAULT_CHUNK_FILE,
        help="Path to the processed chunk JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of chunks to load (use 0 or negative for all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kg_demo.png"),
        help="Where to save the visualization PNG (omit to skip saving).",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving the visualization image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None
    output_path = None if args.no_output else args.output
    build_and_visualize_kg(args.chunk_file, limit, output_path)


if __name__ == "__main__":
    main()
