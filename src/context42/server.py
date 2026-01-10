"""MCP server for Context42."""

from fastmcp import FastMCP

from .config import settings
from .storage import SourceStorage, VectorStorage
from .indexer import Embedder
from .search import SearchEngine


# Initialize FastMCP server
mcp = FastMCP("context42")

# Lazy initialization
_search_engine = None


def get_search_engine() -> SearchEngine:
    """
    Get or initialize search engine (lazy loading).

    Returns:
        SearchEngine instance
    """
    global _search_engine

    if _search_engine is None:
        # Ensure directories exist
        settings.ensure_dirs()

        # Initialize storage
        source_storage = SourceStorage(settings.db_path)
        vector_storage = VectorStorage(settings.vectors_path)
        vector_storage.init_db()

        # Initialize embedder
        embedder = Embedder(settings.embedding_model)

        # Create search engine with source_storage for priority lookup
        _search_engine = SearchEngine(vector_storage, embedder, source_storage)

    return _search_engine


@mcp.tool()
def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search personal coding instructions using vector similarity with priority weighting.

    This tool searches through your indexed coding instructions, guidelines,
    and documentation to find relevant information based on semantic similarity.
    Results are ordered by priority-weighted scores to prioritize personal
    instructions over reference documentation.

    Args:
        query: The search query (e.g., "how to handle errors in Python")
        top_k: Maximum number of results to return (default: 5, max: 20)

    Returns:
        List of relevant text chunks with similarity scores, sorted by priority-weighted relevance.
        Each result contains:
        - text: The content of the chunk
        - source: Name of the source this came from
        - file: Relative path to the file
        - score: Similarity score (0-1, higher is more relevant)
        - priority: Priority of the source (0.1-1.0, higher means more important)
        - is_priority: True if this is a priority source (priority >= 0.8)
    """
    # Validate top_k
    if top_k < 1:
        top_k = 1
    elif top_k > 20:
        top_k = 20

    # Get search engine
    engine = get_search_engine()

    # Perform search
    results = engine.search(query, top_k=top_k)

    # Convert to dict format
    return [
        {
            "text": r.text,
            "source": r.source_name,
            "file": r.file_path,
            "score": round(r.score, 4),
            "priority": round(r.priority, 1),
            "is_priority": r.is_priority,
        }
        for r in results
    ]


def run_server():
    """Entry point for starting the MCP server."""
    mcp.run()
