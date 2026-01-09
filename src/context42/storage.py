"""Storage layer for Context42: SQLite for metadata, LanceDB for vectors."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import uuid

import lancedb


@dataclass
class Source:
    """Source metadata from SQLite."""
    name: str
    path: str
    indexed_at: Optional[str]
    embedding_model: Optional[str]


class SourceStorage:
    """SQLite storage for source metadata with atomic writes."""

    def __init__(self, db_path: Path):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self.init_db()

    def _connect(self) -> None:
        """Open connection to SQLite database and enable WAL mode."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # CRITICAL: Enable WAL mode for atomic writes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    def init_db(self) -> None:
        """Create sources table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                indexed_at TEXT,
                embedding_model TEXT
            )
        """)
        self.conn.commit()

    def add_source(self, name: str, path: str) -> None:
        """
        Add a new source.

        Args:
            name: Unique source name
            path: Path to source directory

        Raises:
            sqlite3.IntegrityError: If source name already exists
        """
        self.conn.execute(
            "INSERT INTO sources (name, path, indexed_at, embedding_model) VALUES (?, ?, NULL, NULL)",
            (name, path)
        )
        self.conn.commit()

    def list_sources(self) -> list[Source]:
        """
        List all sources.

        Returns:
            List of Source objects
        """
        cursor = self.conn.execute("SELECT * FROM sources ORDER BY name")
        rows = cursor.fetchall()
        return [Source(
            name=row["name"],
            path=row["path"],
            indexed_at=row["indexed_at"],
            embedding_model=row["embedding_model"]
        ) for row in rows]

    def get_source(self, name: str) -> Optional[Source]:
        """
        Get a source by name.

        Args:
            name: Source name

        Returns:
            Source object or None if not found
        """
        cursor = self.conn.execute("SELECT * FROM sources WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return Source(
                name=row["name"],
                path=row["path"],
                indexed_at=row["indexed_at"],
                embedding_model=row["embedding_model"]
            )
        return None

    def update_indexed(self, name: str, embedding_model: str) -> None:
        """
        Mark source as indexed with timestamp and model.

        Args:
            name: Source name
            embedding_model: Embedding model used for indexing
        """
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE sources SET indexed_at = ?, embedding_model = ? WHERE name = ?",
            (now, embedding_model, name)
        )
        self.conn.commit()

    def remove_source(self, name: str) -> bool:
        """
        Remove a source.

        Args:
            name: Source name

        Returns:
            True if removed, False if source didn't exist
        """
        cursor = self.conn.execute("DELETE FROM sources WHERE name = ?", (name,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()


class VectorStorage:
    """LanceDB storage for vector embeddings with idempotency."""

    def __init__(self, vectors_path: Path):
        """
        Initialize LanceDB storage.

        Args:
            vectors_path: Path to LanceDB directory
        """
        self.vectors_path = vectors_path
        self.vectors_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.vectors_path))
        self.table_name = "chunks"
        self.table: Optional[lancedb.table.Table] = None

    def init_db(self) -> None:
        """Create or open chunks table."""
        # Check if table exists
        table_names = self.db.table_names()

        if self.table_name in table_names:
            self.table = self.db.open_table(self.table_name)
        else:
            # Create empty table with schema
            # LanceDB will create the table when first data is inserted
            self.table = None

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to vector database (batch insert).

        Args:
            chunks: List of chunk dictionaries with schema:
                {
                    "id": str (UUID),
                    "vector": list[float] (384 dims),
                    "text": str,
                    "source_name": str,
                    "file_path": str,
                    "chunk_index": int,
                    "embedding_model": str,
                    "content_hash": str
                }
        """
        if not chunks:
            return

        # Add UUIDs if not present
        for chunk in chunks:
            if "id" not in chunk:
                chunk["id"] = str(uuid.uuid4())

        # Create table if it doesn't exist, otherwise append
        if self.table is None:
            self.table = self.db.create_table(self.table_name, chunks)
        else:
            self.table.add(chunks)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of result dictionaries with similarity scores
        """
        if self.table is None:
            return []

        results = self.table.search(query_vector).limit(top_k).to_list()
        return results

    def chunk_exists(self, content_hash: str) -> bool:
        """
        Check if a chunk with given content hash already exists.

        Args:
            content_hash: SHA256 hash of chunk content

        Returns:
            True if chunk exists, False otherwise
        """
        if self.table is None:
            return False

        try:
            results = self.table.search().where(f"content_hash = '{content_hash}'").limit(1).to_list()
            return len(results) > 0
        except Exception:
            # If search fails (table empty, etc), assume doesn't exist
            return False

    def delete_by_source(self, source_name: str) -> int:
        """
        Delete all chunks from a source.

        Args:
            source_name: Source name to delete

        Returns:
            Number of chunks deleted
        """
        if self.table is None:
            return 0

        # Count before deleting
        count_before = self.table.count_rows()

        # Delete chunks
        self.table.delete(f"source_name = '{source_name}'")

        # Count after to calculate deleted
        count_after = self.table.count_rows()
        return count_before - count_after

    def get_chunks_by_source(self, source_name: str) -> list[dict]:
        """
        Get all chunks from a source.

        Args:
            source_name: Source name

        Returns:
            List of chunk dictionaries
        """
        if self.table is None:
            return []

        results = self.table.search().where(f"source_name = '{source_name}'").to_list()
        return results

    def get_stats(self) -> dict:
        """
        Get statistics about vector storage.

        Returns:
            Dictionary with:
                - total_chunks: Total number of chunks
                - sources: Dict mapping source_name to chunk count
                - storage_size_bytes: Total storage size in bytes
        """
        if self.table is None:
            return {
                "total_chunks": 0,
                "sources": {},
                "storage_size_bytes": 0,
            }

        # Total chunks
        total = self.table.count_rows()

        # Count per source - convert to pandas and group
        df = self.table.to_pandas()
        sources = df.groupby("source_name").size().to_dict() if not df.empty else {}

        # Storage size - sum all files in vectors directory
        size = sum(
            f.stat().st_size
            for f in self.vectors_path.rglob("*")
            if f.is_file()
        )

        return {
            "total_chunks": total,
            "sources": sources,
            "storage_size_bytes": size,
        }
