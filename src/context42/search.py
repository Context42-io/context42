"""Search engine for vector similarity search."""

from dataclasses import dataclass

from .storage import VectorStorage
from .indexer import Embedder


@dataclass
class SearchResult:
    """Search result with similarity score."""
    text: str
    source_name: str
    file_path: str
    score: float


class SearchEngine:
    """Simple vector search engine."""

    def __init__(self, vector_storage: VectorStorage, embedder: Embedder):
        """
        Initialize search engine.

        Args:
            vector_storage: LanceDB vector storage
            embedder: Embedding generator
        """
        self.vector_storage = vector_storage
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        # Generate embedding for query
        query_embeddings = self.embedder.embed([query])
        query_vector = query_embeddings[0].tolist()

        # Search in vector database
        results = self.vector_storage.search(query_vector, top_k=top_k)

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            # LanceDB returns distance, convert to similarity score
            # Smaller distance = higher similarity
            # We use 1 / (1 + distance) to convert to 0-1 range
            distance = result.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)

            search_results.append(SearchResult(
                text=result.get("text", ""),
                source_name=result.get("source_name", ""),
                file_path=result.get("file_path", ""),
                score=score
            ))

        # Sort by score (highest first)
        search_results.sort(key=lambda x: x.score, reverse=True)

        return search_results
