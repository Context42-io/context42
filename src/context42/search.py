"""Search engine for vector similarity search."""

from dataclasses import dataclass

from .storage import VectorStorage, SourceStorage
from .indexer import Embedder


@dataclass
class SearchResult:
    """Search result with similarity score and priority."""
    text: str
    source_name: str
    file_path: str
    score: float
    priority: float = 1.0
    is_priority: bool = False
    weighted_score: float = 0.0


class SearchEngine:
    """Vector search engine with priority weighting."""

    def __init__(
        self,
        vector_storage: VectorStorage,
        embedder: Embedder,
        source_storage: SourceStorage
    ):
        """
        Initialize search engine.

        Args:
            vector_storage: LanceDB vector storage
            embedder: Embedding generator
            source_storage: SQLite source storage (for priority lookup)
        """
        self.vector_storage = vector_storage
        self.embedder = embedder
        self.source_storage = source_storage
        self._priority_cache: dict[str, float] = {}

    def _get_priority(self, source_name: str) -> float:
        """
        Get priority for a source with caching.

        Args:
            source_name: Source name

        Returns:
            Priority value (defaults to 1.0 if not found)
        """
        if source_name not in self._priority_cache:
            priority = self.source_storage.get_source_priority(source_name)
            self._priority_cache[source_name] = priority if priority is not None else 1.0
        return self._priority_cache[source_name]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for similar chunks using vector similarity with priority weighting.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects, sorted by weighted_score (highest first).
            The 'score' field contains the original vector similarity score,
            while 'weighted_score' contains the priority-weighted score used for ordering.
        """
        # Generate embedding for query
        query_embeddings = self.embedder.embed([query])
        query_vector = query_embeddings[0].tolist()

        # Fetch top_k * 2 results to ensure high-priority items appear after reordering
        fetch_limit = top_k * 2
        results = self.vector_storage.search(query_vector, top_k=fetch_limit)

        # Convert to SearchResult objects with priority weighting
        search_results = []
        for result in results:
            # LanceDB returns distance, convert to similarity score
            # Smaller distance = higher similarity
            # We use 1 / (1 + distance) to convert to 0-1 range
            distance = result.get("_distance", 0.0)
            vector_score = 1.0 / (1.0 + distance)

            # Get priority for this source
            source_name = result.get("source_name", "")
            priority = self._get_priority(source_name)

            # Calculate weighted score for ordering
            weighted_score = vector_score * priority

            # Determine if this is a priority source (threshold: 0.8)
            is_priority = priority >= 0.8

            search_results.append(SearchResult(
                text=result.get("text", ""),
                source_name=source_name,
                file_path=result.get("file_path", ""),
                score=vector_score,  # Original vector score
                priority=priority,
                is_priority=is_priority,
                weighted_score=weighted_score
            ))

        # Sort by weighted_score (highest first)
        search_results.sort(key=lambda x: x.weighted_score, reverse=True)

        # Return top_k results
        return search_results[:top_k]
