"""Indexer: markdown chunking + embedding generation."""

import hashlib
import re
import fnmatch
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

from fastembed import TextEmbedding

from .storage import SourceStorage, VectorStorage
from .config import Settings, DEFAULT_EXCLUDES, SUPPORTED_EXTENSIONS


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


def should_exclude(file_path: Path, source_path: Path, excludes: list[str]) -> bool:
    """
    Check if a file should be excluded from indexing.

    Args:
        file_path: Absolute path to file
        source_path: Root source directory
        excludes: List of exclude patterns

    Returns:
        True if file should be excluded
    """
    # Get relative path for pattern matching
    try:
        relative = file_path.relative_to(source_path)
    except ValueError:
        return False

    relative_str = str(relative)

    # Check each pattern
    for pattern in excludes:
        # Check if any part of the path matches
        parts = relative.parts

        for part in parts:
            # Direct match
            if fnmatch.fnmatch(part, pattern):
                return True

        # Full path match
        if fnmatch.fnmatch(relative_str, pattern):
            return True

        # Pattern with path separator
        if fnmatch.fnmatch(relative_str, f"*/{pattern}"):
            return True
        if fnmatch.fnmatch(relative_str, f"{pattern}/*"):
            return True
        if fnmatch.fnmatch(relative_str, f"*/{pattern}/*"):
            return True

    return False


def get_effective_excludes(source_excludes: Optional[str]) -> list[str]:
    """
    Get effective exclude patterns combining defaults and custom.

    Args:
        source_excludes: JSON string of custom excludes (may contain __NO_DEFAULTS__ marker)

    Returns:
        List of patterns to exclude
    """
    custom = json.loads(source_excludes) if source_excludes else []

    # Check for no-defaults marker
    use_defaults = "__NO_DEFAULTS__" not in custom
    custom = [e for e in custom if e != "__NO_DEFAULTS__"]

    if use_defaults:
        return list(DEFAULT_EXCLUDES) + custom
    else:
        return custom


def chunk_markdown(text: str, chunk_size: int = 500, overlap: int = 50, min_chunk_size: int = 100) -> list[str]:
    """
    Chunk markdown text ensuring each chunk contains meaningful content.

    CRITICAL FIX (v0.2.1): This function was creating header-only chunks that were
    useless for LLM search. Now ensures every chunk has header + real content.

    Algorithm:
    1. Parse markdown into (header, content) pairs
    2. Accumulate small sections until reaching min_chunk_size
    3. Split large sections with header preserved
    4. Validate chunks have sufficient non-header content

    Args:
        text: Markdown text to chunk
        chunk_size: Target chunk size in CHARACTERS
        overlap: Overlap between chunks in CHARACTERS
        min_chunk_size: Minimum content size to create useful chunks

    Returns:
        List of text chunks, each with header + meaningful content
    """
    chunks = []

    # Parse into (header, content) pairs
    sections = _parse_sections(text)

    # Accumulator for small sections
    accumulated_header = ""
    accumulated_content = ""

    for header, content in sections:
        # Skip sections with no real content (empty or whitespace only)
        content_stripped = content.strip()
        if not content_stripped:
            continue

        # If accumulated content + new content fits and stays useful
        total_content = accumulated_content + "\n\n" + content_stripped if accumulated_content else content_stripped

        # If this section is too small on its own, accumulate it
        if len(content_stripped) < min_chunk_size:
            if not accumulated_header:
                accumulated_header = header
            accumulated_content = total_content
            continue

        # Process accumulated content if any
        if accumulated_content:
            full_accumulated = f"{accumulated_header}\n\n{accumulated_content}".strip()
            if len(full_accumulated) > chunk_size:
                # Split large accumulated section
                chunks.extend(_split_large_section(accumulated_header, accumulated_content, chunk_size, overlap))
            else:
                chunks.append(full_accumulated)
            accumulated_header = ""
            accumulated_content = ""

        # Process current section
        full_section = f"{header}\n\n{content_stripped}".strip()
        if len(full_section) > chunk_size:
            # Split large section
            chunks.extend(_split_large_section(header, content_stripped, chunk_size, overlap))
        else:
            chunks.append(full_section)

    # Don't forget last accumulated content
    if accumulated_content:
        full_accumulated = f"{accumulated_header}\n\n{accumulated_content}".strip()
        chunks.append(full_accumulated)

    # Final validation: ensure chunks have meaningful content
    final_chunks = []
    for chunk in chunks:
        if not chunk.strip():
            continue

        # Check that chunk has content beyond just header(s)
        lines = chunk.strip().split('\n')
        non_header_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        non_header_content = '\n'.join(non_header_lines).strip()

        # Require at least some real content (not just headers)
        if len(non_header_content) >= 20:  # Minimum useful content
            final_chunks.append(chunk)
        # Silently skip header-only or near-empty chunks

    return final_chunks


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """
    Parse markdown into (header, content) pairs.

    This separates headers from their content, which is critical for
    preventing header-only chunks.

    Handles header hierarchy: sub-headers (###, ####) inherit their immediate
    parent header (##) for context when the parent has no direct content.

    Args:
        text: Markdown text

    Returns:
        List of (header, content) tuples with headers preserving hierarchy
    """
    header_pattern = r'^(#{1,6}\s+.+)$'
    lines = text.split('\n')

    sections = []
    recent_h2 = ""  # Most recent ## header (for sub-header context)
    current_header = ""
    current_content_lines = []

    for line in lines:
        header_match = re.match(header_pattern, line)
        if header_match:
            # Save previous section
            if current_header or current_content_lines:
                content = '\n'.join(current_content_lines)
                sections.append((current_header, content))

            # Determine header level
            header_level = len(line) - len(line.lstrip('#'))

            # Update recent H2 if this is a level 2 header
            if header_level == 2:
                recent_h2 = line

            # Build header with parent context if this is a sub-header
            if header_level > 2 and recent_h2:
                # Sub-header: include parent ## for context
                full_header = f"{recent_h2}\n\n{line}"
            else:
                # Top-level header or no parent
                full_header = line

            # Start new section
            current_header = full_header
            current_content_lines = []
        else:
            # Content lines (not headers)
            current_content_lines.append(line)

    # Save last section
    if current_header or current_content_lines:
        content = '\n'.join(current_content_lines)
        sections.append((current_header, content))

    return sections


def _split_large_section(header: str, content: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a large section into multiple chunks, each with the header.

    Args:
        header: Section header
        content: Section content (without header)
        chunk_size: Target chunk size
        overlap: Overlap size

    Returns:
        List of chunks, each prefixed with header
    """
    chunks = []

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', content)

    current_chunk_content = ""
    header_prefix = f"{header}\n\n" if header else ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        potential_size = len(header_prefix) + len(current_chunk_content) + len(para) + 2

        if potential_size > chunk_size and current_chunk_content:
            # Save current chunk with header
            chunks.append(f"{header_prefix}{current_chunk_content}".strip())

            # Start new chunk with overlap
            overlap_text = _get_smart_overlap(current_chunk_content, overlap)
            current_chunk_content = overlap_text + "\n\n" + para if overlap_text else para
        else:
            # Add to current chunk
            if current_chunk_content:
                current_chunk_content += "\n\n" + para
            else:
                current_chunk_content = para

    # Save last chunk
    if current_chunk_content:
        chunks.append(f"{header_prefix}{current_chunk_content}".strip())

    return chunks


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


def _parse_rst_sections(text: str) -> list[tuple[str, str]]:
    """
    Parse RST into (header, content) pairs.

    RST uses underlines (and optional overlines) for headers:
    - Characters: = - ` : . ' " ~ ^ _ * + #
    - First style encountered becomes H1, second H2, etc.

    Args:
        text: RST text

    Returns:
        List of (header, content) tuples
    """
    lines = text.split('\n')
    sections = []

    # RST header underline characters
    header_chars = set('=-`:\'.\"~^_*+#')

    # Track header hierarchy (first seen = H1, etc)
    seen_styles: list[str] = []

    current_header = ""
    current_content_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for header pattern
        is_header = False
        header_text = ""
        style = ""

        # Pattern 1: Overline + Title + Underline (e.g., === Title ===)
        if (i + 2 < len(lines) and
            len(line) >= 3 and
            line[0] in header_chars and
            len(set(line.strip())) == 1):

            title_line = lines[i + 1]
            underline = lines[i + 2] if i + 2 < len(lines) else ""

            if (len(underline) >= 3 and
                underline[0] == line[0] and
                len(set(underline.strip())) == 1 and
                len(title_line.strip()) > 0):

                is_header = True
                header_text = title_line.strip()
                style = line[0] + "_overline"

                if style not in seen_styles:
                    seen_styles.append(style)

                i += 3

        # Pattern 2: Title + Underline only
        if not is_header and i + 1 < len(lines):
            next_line = lines[i + 1]

            if (len(line.strip()) > 0 and
                len(next_line) >= 3 and
                next_line[0] in header_chars and
                len(set(next_line.strip())) == 1 and
                len(next_line.strip()) >= len(line.strip())):

                is_header = True
                header_text = line.strip()
                style = next_line[0]

                if style not in seen_styles:
                    seen_styles.append(style)

                i += 2

        if is_header:
            # Save previous section
            if current_header or current_content_lines:
                content = '\n'.join(current_content_lines)
                sections.append((current_header, content))

            # Get header level based on order seen
            level = seen_styles.index(style) + 1
            header_prefix = "#" * min(level, 6)  # Convert to markdown-style for consistency

            current_header = f"{header_prefix} {header_text}"
            current_content_lines = []
        else:
            # Regular content line
            current_content_lines.append(line)
            i += 1

    # Save last section
    if current_header or current_content_lines:
        content = '\n'.join(current_content_lines)
        sections.append((current_header, content))

    return sections


def _clean_rst_content(content: str) -> str:
    """
    Clean RST-specific formatting for better embedding.

    Args:
        content: Raw RST content

    Returns:
        Cleaned content
    """
    lines = content.split('\n')
    cleaned = []

    in_code_block = False

    for line in lines:
        # Handle code blocks (.. code-block::)
        if line.strip().startswith('.. code-block::'):
            in_code_block = True
            cleaned.append('')  # Blank line before code
            cleaned.append('```')
            continue

        # Handle other directives (skip them)
        if line.strip().startswith('.. '):
            # Common directives to skip: note, warning, seealso, etc
            if any(d in line for d in ['note::', 'warning::', 'seealso::', 'versionadded::',
                                         'versionchanged::', 'deprecated::', 'todo::']):
                continue

        # End code block on dedent
        if in_code_block and line and not line.startswith(' ') and not line.startswith('\t'):
            cleaned.append('```')
            cleaned.append('')
            in_code_block = False

        # Convert RST inline formatting to markdown-ish
        line = line.replace('``', '`')  # Inline code
        line = line.replace('::', ':')   # Literal blocks

        cleaned.append(line)

    # Close any open code block
    if in_code_block:
        cleaned.append('```')

    return '\n'.join(cleaned)


def chunk_rst(text: str, chunk_size: int = 500, overlap: int = 50, min_chunk_size: int = 100) -> list[str]:
    """
    Chunk RST text ensuring each chunk contains meaningful content.

    Converts RST sections to a normalized format and applies same
    chunking logic as markdown.

    Args:
        text: RST text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        min_chunk_size: Minimum content size for useful chunks

    Returns:
        List of text chunks
    """
    chunks = []

    # Parse into (header, content) pairs
    sections = _parse_rst_sections(text)

    # Accumulator for small sections
    accumulated_header = ""
    accumulated_content = ""

    for header, content in sections:
        # Clean up RST-specific formatting
        content = _clean_rst_content(content)

        content_stripped = content.strip()
        if not content_stripped:
            continue

        total_content = accumulated_content + "\n\n" + content_stripped if accumulated_content else content_stripped

        if len(content_stripped) < min_chunk_size:
            if not accumulated_header:
                accumulated_header = header
            accumulated_content = total_content
            continue

        # Process accumulated content
        if accumulated_content:
            full_accumulated = f"{accumulated_header}\n\n{accumulated_content}".strip()
            if len(full_accumulated) > chunk_size:
                chunks.extend(_split_large_section(accumulated_header, accumulated_content, chunk_size, overlap))
            else:
                chunks.append(full_accumulated)
            accumulated_header = ""
            accumulated_content = ""

        # Process current section
        full_section = f"{header}\n\n{content_stripped}".strip()
        if len(full_section) > chunk_size:
            chunks.extend(_split_large_section(header, content_stripped, chunk_size, overlap))
        else:
            chunks.append(full_section)

    # Last accumulated
    if accumulated_content:
        full_accumulated = f"{accumulated_header}\n\n{accumulated_content}".strip()
        chunks.append(full_accumulated)

    # Final validation
    final_chunks = []
    for chunk in chunks:
        if not chunk.strip():
            continue

        lines = chunk.strip().split('\n')
        non_header_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        non_header_content = '\n'.join(non_header_lines).strip()

        if len(non_header_content) >= 20:
            final_chunks.append(chunk)

    return final_chunks


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
            with console.status(
                f"[yellow]Loading model: {self._model_name}[/yellow]",
                spinner="dots"
            ):
                self._model = TextEmbedding(model_name=self._model_name)
            console.print("[green]✓ Model loaded[/green]")

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

    def _print_summary(
        self,
        source_name: str,
        is_first_index: bool,
        skipped_chunks: int,
        total_indexed: int,
        total_batches: int,
        total_found: int
    ) -> None:
        """
        Print indexing summary with appropriate warnings.

        Args:
            source_name: Source name
            is_first_index: Whether this is the first time indexing this source
            skipped_chunks: Number of chunks skipped (already existed)
            total_indexed: Number of chunks indexed
            total_batches: Number of batches processed
            total_found: Total chunks found (skipped + indexed)
        """
        console.print()  # Blank line after progress bars

        # Detect ghost source situation
        if is_first_index and skipped_chunks > 0 and total_indexed == 0:
            console.print(f"[yellow]⚠️  Warning: All {skipped_chunks} chunks already exist in other sources.[/yellow]")
            console.print(f"[yellow]   Source '{source_name}' has no unique content to search.[/yellow]")
            console.print()
            console.print("[dim]   This may happen when:[/dim]")
            console.print("[dim]   - The same directory is indexed under different names[/dim]")
            console.print("[dim]   - Subdirectories overlap with existing sources[/dim]")
            console.print()
            console.print("[dim]   Consider:[/dim]")
            console.print(f"[dim]   - Remove this source: c42 remove {source_name}[/dim]")
            console.print("[dim]   - Or adjust priority of existing source with same content[/dim]")
            console.print()
            console.print(f"[yellow]✓ Indexing complete: 0 chunks created[/yellow]")
        elif skipped_chunks > 0 and total_indexed > 0:
            # Some new, some skipped - normal partial update
            console.print(f"[dim]Skipped {skipped_chunks} already indexed chunks[/dim]")
            console.print(f"[green]✓ Indexed {total_indexed} chunks in {total_batches} batch(es)[/green]")
        elif skipped_chunks > 0 and not is_first_index:
            # Re-index - all already exist
            console.print(f"[dim]Skipped {skipped_chunks} already indexed chunks[/dim]")
            console.print(f"[green]✓ All chunks already indexed (idempotent)[/green]")
        elif total_indexed > 0:
            # Fresh index
            console.print(f"[green]✓ Indexed {total_indexed} chunks in {total_batches} batch(es)[/green]")
        elif total_found == 0:
            console.print("[yellow]No chunks found to index (empty or invalid files)[/yellow]")
        else:
            console.print(f"[green]✓ All chunks already indexed (idempotent)[/green]")

    def index_source(self, source_name: str) -> int:
        """
        Index a single source using two-phase processing with progress bars.

        Phase 1: Read files and chunk (accumulate all chunks)
        Phase 2: Embed and insert (in batches with progress bar)

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

        # Check if first index (for warning logic)
        is_first_index = source.indexed_at is None

        # Check model compatibility
        self.check_model_compatibility(source_name)

        console.print(f"\n[bold blue]Indexing source: {source_name}[/bold blue]")
        console.print(f"[dim]Path: {source.path}[/dim]")

        # Get source path
        source_path = Path(source.path).expanduser()
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source.path}")

        # Get effective excludes
        excludes = get_effective_excludes(source.excludes)

        # Find all supported files (.md, .rst)
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(source_path.rglob(f"*{ext}"))

        # Filter out excluded files
        files_to_index = [
            f for f in all_files
            if not should_exclude(f, source_path, excludes)
        ]

        console.print(f"[blue]Found {len(files_to_index)} files to index[/blue]")
        if len(all_files) > len(files_to_index):
            excluded_count = len(all_files) - len(files_to_index)
            console.print(f"[dim]Excluded {excluded_count} files by patterns[/dim]")

        if not files_to_index:
            console.print("[yellow]No files found to index[/yellow]")
            return 0

        # Pre-load embedding model (shows loading message before progress bars)
        self.embedder._ensure_model()

        console.print()  # Blank line

        # =========================================
        # PHASE 1: Read and chunk all files
        # =========================================
        chunks_to_embed: list[dict] = []
        skipped_chunks = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,  # Keep visible after completion
        ) as progress:

            read_task = progress.add_task(
                "[cyan]Reading files",
                total=len(files_to_index)
            )

            for file_path in files_to_index:
                # Read file
                try:
                    content = file_path.read_text(encoding='utf-8')
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
                    progress.advance(read_task)
                    continue

                # Choose chunker based on extension
                if file_path.suffix.lower() == ".rst":
                    # RST chunker
                    file_chunks = chunk_rst(
                        content,
                        self.settings.chunk_size,
                        self.settings.chunk_overlap
                    )
                else:
                    # Markdown chunker
                    file_chunks = chunk_markdown(
                        content,
                        self.settings.chunk_size,
                        self.settings.chunk_overlap
                    )

                # Prepare chunks with metadata
                relative_path = file_path.relative_to(source_path)

                for idx, chunk_text in enumerate(file_chunks):
                    content_hash = compute_content_hash(chunk_text)

                    # Check if already exists (idempotency)
                    if self.vector_storage.chunk_exists(content_hash):
                        skipped_chunks += 1
                        continue

                    chunks_to_embed.append({
                        "text": chunk_text,
                        "source_name": source_name,
                        "file_path": str(relative_path),
                        "chunk_index": idx,
                        "embedding_model": self.embedder.model_name,
                        "content_hash": content_hash
                    })

                progress.advance(read_task)

        # Check if anything to embed
        total_chunks_found = skipped_chunks + len(chunks_to_embed)

        if not chunks_to_embed:
            # Handle ghost source warning or normal re-index
            self._print_summary(
                source_name=source_name,
                is_first_index=is_first_index,
                skipped_chunks=skipped_chunks,
                total_indexed=0,
                total_batches=0,
                total_found=total_chunks_found
            )
            self.source_storage.update_indexed(source_name, self.embedder.model_name)
            return 0

        # =========================================
        # PHASE 2: Embed and insert in batches
        # =========================================
        total_chunks = len(chunks_to_embed)
        total_batches = (total_chunks + self.settings.batch_size - 1) // self.settings.batch_size
        total_indexed = 0
        batch_num = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            embed_task = progress.add_task(
                "[cyan]Embedding chunks",
                total=total_chunks
            )

            # Process in batches
            for i in range(0, total_chunks, self.settings.batch_size):
                batch = chunks_to_embed[i:i + self.settings.batch_size]
                batch_num += 1

                # Generate embeddings
                texts = [chunk["text"] for chunk in batch]
                embeddings = self.embedder.embed(texts)

                # Add vectors to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk["vector"] = embedding.tolist()

                # Insert to LanceDB
                self.vector_storage.add_chunks(batch)

                total_indexed += len(batch)
                progress.advance(embed_task, advance=len(batch))

        # Update source as indexed
        self.source_storage.update_indexed(source_name, self.embedder.model_name)

        # Print summary
        self._print_summary(
            source_name=source_name,
            is_first_index=is_first_index,
            skipped_chunks=skipped_chunks,
            total_indexed=total_indexed,
            total_batches=batch_num,
            total_found=total_chunks_found
        )

        return total_indexed
