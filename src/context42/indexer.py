"""Indexer: markdown chunking + embedding generation."""

import hashlib
import re
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fastembed import TextEmbedding

from .storage import SourceStorage, VectorStorage
from .config import Settings


console = Console()


def compute_content_hash(text: str) -> str:
    """
    Compute SHA256 hash of text for idempotency.

    Args:
        text: Text to hash

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_markdown(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Chunk markdown text while preserving semantic context.

    Algorithm:
    1. Split by headers (##, ###, etc)
    2. If section > chunk_size:
       a. Split by paragraphs (blank lines)
       b. If paragraph > chunk_size:
          c. Split by sentences (., !, ?)
          d. NEVER split mid-sentence
    3. Apply overlap between adjacent chunks
    4. Preserve context: include parent header at chunk start

    Args:
        text: Markdown text to chunk
        chunk_size: Target chunk size in CHARACTERS
        overlap: Overlap between chunks in CHARACTERS

    Returns:
        List of text chunks
    """
    chunks = []

    # Split by headers first
    header_pattern = r'^(#{1,6}\s+.+)$'
    lines = text.split('\n')

    current_header = ""
    current_section = []

    for line in lines:
        if re.match(header_pattern, line):
            # Process previous section if exists
            if current_section:
                section_text = '\n'.join(current_section)
                chunks.extend(_chunk_section(section_text, current_header, chunk_size, overlap))

            # Start new section
            current_header = line
            current_section = [line]
        else:
            current_section.append(line)

    # Process last section
    if current_section:
        section_text = '\n'.join(current_section)
        chunks.extend(_chunk_section(section_text, current_header, chunk_size, overlap))

    return [c for c in chunks if c.strip()]


def _chunk_section(section_text: str, header: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk a single section of text."""
    if len(section_text) <= chunk_size:
        return [section_text]

    chunks = []

    # Split by paragraphs (blank lines)
    paragraphs = re.split(r'\n\s*\n', section_text)

    current_chunk = header + "\n\n" if header else ""
    header_prefix = header + "\n\n" if header else ""

    for para in paragraphs:
        # If adding paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size and current_chunk.strip():
            # Save current chunk
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap (find word boundary)
            overlap_text = _get_smart_overlap(current_chunk, overlap)
            current_chunk = header_prefix + overlap_text + "\n\n"

        # If paragraph itself is too large, split by sentences
        if len(para) > chunk_size:
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(current_chunk) + len(sent) > chunk_size and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    overlap_text = _get_smart_overlap(current_chunk, overlap)
                    current_chunk = header_prefix + overlap_text + " "

                current_chunk += sent + " "
        else:
            current_chunk += para + "\n\n"

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _get_smart_overlap(text: str, overlap: int) -> str:
    """
    Get overlap text without cutting mid-word.

    Finds the nearest word boundary within the overlap region.
    """
    if len(text) <= overlap:
        return text.lstrip()

    # Get target overlap region
    target_start = len(text) - overlap

    # Find nearest space or sentence boundary before target
    search_text = text[max(0, target_start - 20):target_start + overlap]

    # Look for sentence boundaries first (., !, ?, newline)
    for boundary in ['. ', '! ', '? ', '\n']:
        pos = search_text.rfind(boundary)
        if pos != -1:
            actual_start = max(0, target_start - 20) + pos + len(boundary)
            return text[actual_start:].lstrip()

    # Fall back to word boundary (space)
    pos = search_text.rfind(' ')
    if pos != -1:
        actual_start = max(0, target_start - 20) + pos + 1
        return text[actual_start:].lstrip()

    # If no boundary found, return from target (shouldn't happen often)
    return text[target_start:].lstrip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (never mid-sentence)."""
    # Simple sentence splitter (handles ., !, ?)
    sentence_pattern = r'([.!?]+\s+|\n)'
    parts = re.split(sentence_pattern, text)

    sentences = []
    current = ""

    for i, part in enumerate(parts):
        current += part
        # If this is a sentence boundary (odd index in split result)
        if i % 2 == 1:
            sentences.append(current.strip())
            current = ""

    # Add remaining text if any
    if current.strip():
        sentences.append(current.strip())

    return [s for s in sentences if s.strip()]


class Embedder:
    """Embedding generator using fastembed (ONNX, offline)."""

    def __init__(self, model_name: str):
        """
        Initialize embedder with lazy loading.

        Args:
            model_name: Embedding model name
        """
        self._model_name = model_name
        self._model: Optional[TextEmbedding] = None

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def _ensure_model(self) -> None:
        """Lazy load model on first use."""
        if self._model is None:
            console.print(f"[yellow]Loading embedding model: {self._model_name}...[/yellow]")
            self._model = TextEmbedding(model_name=self._model_name)
            console.print("[green]Model loaded successfully[/green]")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._ensure_model()

        # Convert generator to list
        embeddings_generator = self._model.embed(texts)
        embeddings = list(embeddings_generator)

        return embeddings


class Indexer:
    """Main indexer class that orchestrates chunking and embedding."""

    def __init__(
        self,
        source_storage: SourceStorage,
        vector_storage: VectorStorage,
        embedder: Embedder,
        settings: Settings
    ):
        """
        Initialize indexer.

        Args:
            source_storage: SQLite storage for sources
            vector_storage: LanceDB storage for vectors
            embedder: Embedding generator
            settings: Application settings
        """
        self.source_storage = source_storage
        self.vector_storage = vector_storage
        self.embedder = embedder
        self.settings = settings

    def check_model_compatibility(self, source_name: str) -> bool:
        """
        Check if embedding model changed for a source.

        Args:
            source_name: Source name

        Returns:
            True if compatible, False if model mismatch
        """
        source = self.source_storage.get_source(source_name)
        if not source or not source.embedding_model:
            return True

        if source.embedding_model != self.embedder.model_name:
            console.print(
                f"[yellow]Warning: Source '{source_name}' was indexed with "
                f"'{source.embedding_model}', but current model is '{self.embedder.model_name}'.\n"
                f"Results may be incorrect. Consider removing and re-adding the source.[/yellow]"
            )
            return False

        return True

    def index_source(self, source_name: str) -> int:
        """
        Index a single source.

        Args:
            source_name: Source name to index

        Returns:
            Number of chunks created

        Raises:
            ValueError: If source not found
        """
        # Get source
        source = self.source_storage.get_source(source_name)
        if not source:
            raise ValueError(f"Source '{source_name}' not found")

        # Check model compatibility
        self.check_model_compatibility(source_name)

        console.print(f"[blue]Indexing source: {source_name}[/blue]")
        console.print(f"[blue]Path: {source.path}[/blue]")

        # Find all .md files
        source_path = Path(source.path).expanduser()
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source.path}")

        md_files = list(source_path.rglob("*.md"))
        console.print(f"[blue]Found {len(md_files)} markdown files[/blue]")

        if not md_files:
            console.print("[yellow]No markdown files found[/yellow]")
            return 0

        # Process files
        total_chunks = 0
        chunks_to_embed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(md_files))

            for md_file in md_files:
                progress.update(task, description=f"Processing: {md_file.name}")

                # Read file
                try:
                    content = md_file.read_text(encoding='utf-8')
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not read {md_file}: {e}[/yellow]")
                    progress.advance(task)
                    continue

                # Chunk
                file_chunks = chunk_markdown(
                    content,
                    self.settings.chunk_size,
                    self.settings.chunk_overlap
                )

                # Prepare chunks with metadata
                relative_path = md_file.relative_to(source_path)
                for idx, chunk_text in enumerate(file_chunks):
                    content_hash = compute_content_hash(chunk_text)

                    # Check if already exists (idempotency)
                    if self.vector_storage.chunk_exists(content_hash):
                        continue

                    chunks_to_embed.append({
                        "text": chunk_text,
                        "source_name": source_name,
                        "file_path": str(relative_path),
                        "chunk_index": idx,
                        "embedding_model": self.embedder.model_name,
                        "content_hash": content_hash
                    })

                progress.advance(task)

        if not chunks_to_embed:
            console.print("[green]All chunks already indexed (idempotent)[/green]")
            # Update indexed timestamp anyway
            self.source_storage.update_indexed(source_name, self.embedder.model_name)
            return 0

        console.print(f"[blue]Embedding {len(chunks_to_embed)} new chunks...[/blue]")

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks_to_embed]
        embeddings = self.embedder.embed(texts)

        # Add vectors to chunks
        for chunk, embedding in zip(chunks_to_embed, embeddings):
            chunk["vector"] = embedding.tolist()

        # Insert to LanceDB
        console.print("[blue]Storing vectors...[/blue]")
        self.vector_storage.add_chunks(chunks_to_embed)

        # Update source as indexed
        self.source_storage.update_indexed(source_name, self.embedder.model_name)

        total_chunks = len(chunks_to_embed)
        console.print(f"[green]✓ Indexed {total_chunks} chunks[/green]")

        return total_chunks
