"""
Build a Chroma vector index from the knowledge base markdown files.

Usage:
    python my_build_index.py                      # build index
    python my_build_index.py --query "your query" # search after building
"""

import argparse
import logging
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "Task2" / "knowledge_base"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ~500–1000 tokens ≈ 700–1400 characters for English text (1 token ≈ ~4 chars).
# chunk_size=800 chars gives ~200 tokens, comfortably within 100–300 word range.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_documents(kb_dir: Path) -> list[Document]:
    """Read every .md file and return a list of LangChain Documents."""
    documents: list[Document] = []
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    for filepath in md_files:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            log.warning("Skipping empty file: %s", filepath.name)
            continue

        # Use the first non-empty line as the document title
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.strip()), filepath.stem)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(filepath.relative_to(kb_dir.parent.parent)),
                    "filename": filepath.name,
                    "title": title,
                },
            )
        )

    log.info("Loaded %d documents from %s", len(documents), kb_dir)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks, preserving source metadata and adding chunk position."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_total"] = len(doc_chunks)
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunks.extend(doc_chunks)

    log.info("Split into %d chunks (avg %.0f chars)", len(chunks), sum(len(c.page_content) for c in chunks) / max(len(chunks), 1))
    return chunks


def build_index(chunks: list[Document], embedding_model: SentenceTransformerEmbeddings) -> Chroma:
    """Embed chunks and persist them in a Chroma collection.

    Always starts fresh: deletes any existing collection so documents
    are never duplicated across multiple runs.
    """
    if CHROMA_PERSIST_DIR.exists():
        log.info("Removing existing index at %s", CHROMA_PERSIST_DIR)
        shutil.rmtree(CHROMA_PERSIST_DIR)

    log.info("Building Chroma index at %s …", CHROMA_PERSIST_DIR)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name="knowledge_base",
    )
    log.info("Index built: %d vectors stored.", vector_store._collection.count())
    return vector_store


def load_index(embedding_model: SentenceTransformerEmbeddings) -> Chroma:
    """Load an existing Chroma index from disk."""
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model,
        collection_name="knowledge_base",
    )


def search(vector_store: Chroma, query: str, k: int = 5) -> None:
    """Print the top-k most relevant chunks for *query*."""
    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\nTop {k} results for: \"{query}\"\n{'─' * 60}")
    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        print(
            f"[{rank}] score={score:.4f}  |  {m.get('title', '?')}  "
            f"(chunk {m.get('chunk_index', '?')}/{m.get('chunk_total', '?') - 1})\n"
            f"    source : {m.get('source', '?')}\n"
            f"    chunk_id: {m.get('chunk_id', '?')}\n"
            f"    preview : {doc.page_content[:200].replace(chr(10), ' ')!r}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the knowledge-base vector index.")
    parser.add_argument("--query", "-q", type=str, default=None, help="Run a similarity search after building.")
    parser.add_argument("--skip-build", action="store_true", help="Skip (re)building – load existing index instead.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return (default: 5).")
    args = parser.parse_args()

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embedding_model = SentenceTransformerEmbeddings(EMBEDDING_MODEL)

    if args.skip_build and CHROMA_PERSIST_DIR.exists():
        vector_store = load_index(embedding_model)
        log.info("Loaded existing index (%d vectors).", vector_store._collection.count())
    else:
        documents = load_documents(KNOWLEDGE_BASE_DIR)
        chunks = split_documents(documents)
        vector_store = build_index(chunks, embedding_model)

    if args.query:
        search(vector_store, args.query, k=args.top_k)
    else:
        log.info("Done. Use --query 'your question' to search the index.")


if __name__ == "__main__":
    main()
