"""CLI commands for Context42."""

from pathlib import Path
from typing import Annotated, Optional
from functools import wraps
import typer
from rich.console import Console
from rich.table import Table
import sqlite3

from .config import settings
from .storage import SourceStorage, VectorStorage
from .indexer import Embedder, Indexer


app = typer.Typer(
    name="context42",
    help="MCP server for personal coding instructions",
    add_completion=False,
)
console = Console()


def handle_errors(func):
    """Decorator for friendly error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise  # Re-raise Exit for Typer to handle
        except FileNotFoundError as e:
            console.print(f"[red]Error: File not found - {e.filename}[/red]")
            raise typer.Exit(1)
        except PermissionError as e:
            console.print(f"[red]Error: Permission denied - {e.filename}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Run with --verbose for details.[/dim]")
            raise typer.Exit(1)
    return wrapper


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Context42 - MCP server for personal coding instructions."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if ctx.invoked_subcommand is None:
        console.print("[yellow]Context42 - MCP server for personal coding instructions[/yellow]")
        console.print("\nUse --help for available commands")
        console.print("\nQuick start:")
        console.print("  c42 add <path> --name <name>  # Add source directory")
        console.print("  c42 list                      # List sources")
        console.print("  c42 index                     # Index sources")
        console.print("  c42 serve                     # Start MCP server")


@app.command()
@handle_errors
def add(
    path: str = typer.Argument(..., help="Path to source directory"),
    name: str = typer.Option(..., "--name", "-n", help="Unique source name"),
    priority: Annotated[float, typer.Option("--priority", "-p", help="Priority for search results (0.1-1.0, default: 1.0)")] = 1.0,
    exclude: Annotated[Optional[list[str]], typer.Option("--exclude", "-e", help="Additional patterns to exclude (can be used multiple times)")] = None,
    no_default_excludes: Annotated[bool, typer.Option("--no-default-excludes", help="Don't use default exclude patterns")] = False,
):
    """Add a source directory containing markdown or RST files."""
    # Validate priority
    if not 0.1 <= priority <= 1.0:
        console.print(f"[red]Error: Priority must be between 0.1 and 1.0, got {priority}[/red]")
        raise typer.Exit(1)

    # Validate path exists
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    if not source_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(1)

    # Initialize storage
    settings.ensure_dirs()
    storage = SourceStorage(settings.db_path)

    # Check if name already exists
    existing = storage.get_source(name)
    if existing:
        console.print(f"[red]Error: Source '{name}' already exists[/red]")
        console.print(f"[yellow]Existing path: {existing.path}[/yellow]")
        raise typer.Exit(1)

    # Check if path already indexed by another source
    existing_source = storage.get_source_by_path(str(source_path))
    if existing_source:
        console.print(f"[red]Error: Path '{source_path}' is already indexed by source '{existing_source.name}'.[/red]")
        console.print()
        console.print("[dim]Consider:[/dim]")
        console.print(f"[dim]  - Change priority: c42 set-priority {existing_source.name} <value>[/dim]")
        console.print(f"[dim]  - Remove existing: c42 remove {existing_source.name}[/dim]")
        raise typer.Exit(1)

    # Build excludes list
    custom_excludes = exclude if exclude else []

    # Store whether to use defaults (we'll handle this in indexer)
    # If no_default_excludes, we store a marker to indicate no defaults
    if no_default_excludes:
        # Store marker to indicate no defaults
        custom_excludes.insert(0, "__NO_DEFAULTS__")

    # Add source
    try:
        storage.add_source(
            name=name,
            path=str(source_path),
            priority=priority,
            excludes=custom_excludes if custom_excludes else None
        )
        console.print(f"[green]✓ Source '{name}' added successfully[/green]")
        console.print(f"[blue]Path: {source_path}[/blue]")
        console.print(f"[blue]Priority: {priority}[/blue]")

        # Show exclude info
        if custom_excludes:
            display_excludes = [e for e in custom_excludes if e != "__NO_DEFAULTS__"]
            if display_excludes:
                console.print(f"[blue]Custom excludes: {', '.join(display_excludes)}[/blue]")
            if no_default_excludes:
                console.print("[yellow]⚠ Default excludes disabled[/yellow]")
        else:
            console.print("[dim]Using default exclude patterns[/dim]")

        console.print("\n[yellow]Next: Run 'c42 index' to index this source[/yellow]")
    except sqlite3.IntegrityError:
        console.print(f"[red]Error: Source '{name}' already exists[/red]")
        raise typer.Exit(1)
    finally:
        storage.close()


@app.command()
@handle_errors
def list():
    """List all sources and their status."""
    settings.ensure_dirs()
    storage = SourceStorage(settings.db_path)

    try:
        sources = storage.list_sources()

        if not sources:
            console.print("[yellow]No sources configured[/yellow]")
            console.print("\nAdd a source with:")
            console.print("  c42 add <path> --name <name>")
            return

        # Create table
        table = Table(title="Sources")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Path", style="blue")
        table.add_column("Priority", style="yellow", justify="right")
        table.add_column("Indexed", style="green")
        table.add_column("Model", style="magenta")

        for source in sources:
            indexed_status = source.indexed_at if source.indexed_at else "[yellow]pending[/yellow]"
            model_display = source.embedding_model if source.embedding_model else "-"
            priority_display = f"{source.priority:.1f}"

            table.add_row(
                source.name,
                source.path,
                priority_display,
                indexed_status,
                model_display
            )

        console.print(table)

        # Show pending count
        pending = [s for s in sources if not s.indexed_at]
        if pending:
            console.print(f"\n[yellow]{len(pending)} source(s) pending indexing[/yellow]")
            console.print("Run 'c42 index' to index them")

    finally:
        storage.close()


@app.command()
@handle_errors
def remove(
    name: Annotated[str, typer.Argument(help="Name of the source to remove")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
):
    """Remove a source and all its indexed chunks."""
    settings.ensure_dirs()
    source_storage = SourceStorage(settings.db_path)

    try:
        # Check if source exists
        source = source_storage.get_source(name)
        if not source:
            console.print(f"[red]Error: Source '{name}' not found.[/red]")
            console.print("\nAvailable sources:")
            sources = source_storage.list_sources()
            for s in sources:
                console.print(f"  - {s.name}")
            raise typer.Exit(1)

        # Confirmation
        if not yes:
            confirm = typer.confirm(
                f"Remove source '{name}' and all its chunks?",
                default=False
            )
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                raise typer.Exit(0)

        # Remove chunks from LanceDB
        vector_storage = VectorStorage(settings.vectors_path)
        vector_storage.init_db()
        chunks_removed = vector_storage.delete_by_source(name)

        # Remove from SQLite
        source_storage.remove_source(name)

        console.print(f"[green]✓ Removed source '{name}' ({chunks_removed} chunks deleted)[/green]")

    finally:
        source_storage.close()


@app.command()
@handle_errors
def set_priority(
    name: Annotated[str, typer.Argument(help="Name of the source")],
    priority: Annotated[float, typer.Argument(help="New priority value (0.1-1.0)")],
):
    """Update priority for an existing source."""
    # Validate priority
    if not 0.1 <= priority <= 1.0:
        console.print(f"[red]Error: Priority must be between 0.1 and 1.0, got {priority}[/red]")
        raise typer.Exit(1)

    settings.ensure_dirs()
    storage = SourceStorage(settings.db_path)

    try:
        # Check if source exists
        source = storage.get_source(name)
        if not source:
            console.print(f"[red]Error: Source '{name}' not found.[/red]")
            console.print("\nAvailable sources:")
            sources = storage.list_sources()
            for s in sources:
                console.print(f"  - {s.name}")
            raise typer.Exit(1)

        # Update priority
        old_priority = source.priority
        storage.set_priority(name, priority)
        console.print(f"[green]✓ Priority updated for '{name}'[/green]")
        console.print(f"[blue]Old priority: {old_priority}[/blue]")
        console.print(f"[blue]New priority: {priority}[/blue]")

    finally:
        storage.close()


@app.command()
@handle_errors
def status():
    """Show detailed statistics about indexed sources."""
    settings.ensure_dirs()
    source_storage = SourceStorage(settings.db_path)

    try:
        sources = source_storage.list_sources()
        if not sources:
            console.print("[yellow]No sources configured.[/yellow]")
            console.print("Run 'c42 add <path> --name <n>' to add a source.")
            raise typer.Exit(0)

        # Get vector storage statistics
        vector_storage = VectorStorage(settings.vectors_path)
        vector_storage.init_db()

        try:
            stats = vector_storage.get_stats()
        except Exception:
            # Table might not exist yet
            stats = {"total_chunks": 0, "sources": {}, "storage_size_bytes": 0}

        # Header
        console.print("\n[bold]Context42 Status[/bold]\n")

        # Sources summary
        indexed = [s for s in sources if s.indexed_at]
        pending = [s for s in sources if not s.indexed_at]

        console.print(f"[cyan]Sources:[/cyan] {len(sources)} total, {len(indexed)} indexed, {len(pending)} pending")
        console.print(f"[cyan]Chunks:[/cyan]  {stats['total_chunks']:,}")
        console.print(f"[cyan]Storage:[/cyan] {_format_size(stats['storage_size_bytes'])}")
        console.print(f"[cyan]Model:[/cyan]   {settings.embedding_model}")

        # Chunks per source
        if stats["sources"]:
            console.print("\n[bold]Chunks per Source:[/bold]")
            table = Table(show_header=True)
            table.add_column("Source", style="cyan")
            table.add_column("Chunks", justify="right")

            for source_name, chunk_count in sorted(stats["sources"].items()):
                table.add_row(source_name, f"{chunk_count:,}")

            console.print(table)

        # Pending sources
        if pending:
            console.print("\n[yellow]Pending indexing:[/yellow]")
            for s in pending:
                console.print(f"  - {s.name}")
            console.print("\nRun 'c42 index' to index pending sources.")

    finally:
        source_storage.close()


@app.command()
@handle_errors
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results")] = 5,
    source: Annotated[Optional[str], typer.Option("--source", "-s", help="Filter by source")] = None,
):
    """Search indexed instructions (for testing/debugging)."""
    settings.ensure_dirs()

    # Check for indexed sources
    source_storage = SourceStorage(settings.db_path)

    try:
        sources = source_storage.list_sources()
        indexed = [s for s in sources if s.indexed_at]

        if not indexed:
            console.print("[red]Error: No indexed sources.[/red]")
            console.print("Run 'c42 add' and 'c42 index' first.")
            raise typer.Exit(1)

        # Initialize search engine
        from .search import SearchEngine

        vector_storage = VectorStorage(settings.vectors_path)
        vector_storage.init_db()
        embedder = Embedder(settings.embedding_model)
        engine = SearchEngine(vector_storage, embedder, source_storage)

        # Execute search
        console.print(f"[dim]Searching for: {query}[/dim]\n")

        results = engine.search(query, top_k=top_k)

        # Filter by source if specified
        if source:
            results = [r for r in results if r.source_name == source]

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            raise typer.Exit(0)

        # Display results
        for i, r in enumerate(results, 1):
            # Priority indicator
            priority_indicator = "⭐ " if r.is_priority else "  "

            # Color score based on value
            score_color = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.6 else "red"
            console.print(f"[bold]#{i}[/bold] {priority_indicator}[{score_color}]Score: {r.score:.4f}[/{score_color}]")
            console.print(f"    [cyan]Source:[/cyan]   {r.source_name}")
            console.print(f"    [cyan]File:[/cyan]     {r.file_path}")
            console.print(f"    [cyan]Priority:[/cyan] {r.priority:.1f} [dim](weighted: {r.weighted_score:.4f})[/dim]")

            # Truncated text preview
            text_preview = r.text[:200].replace("\n", " ")
            if len(r.text) > 200:
                text_preview += "..."
            console.print(f"    [dim]{text_preview}[/dim]")
            console.print()

    finally:
        source_storage.close()


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@app.command()
@handle_errors
def index():
    """Index all pending sources (creates vector embeddings)."""
    settings.ensure_dirs()
    source_storage = SourceStorage(settings.db_path)
    vector_storage = VectorStorage(settings.vectors_path)
    vector_storage.init_db()

    try:
        # Get pending sources
        sources = source_storage.list_sources()
        pending = [s for s in sources if not s.indexed_at]

        if not pending:
            console.print("[yellow]No sources pending indexing[/yellow]")
            console.print("\nAdd a source with:")
            console.print("  c42 add <path> --name <name>")
            return

        console.print(f"[blue]Found {len(pending)} source(s) to index[/blue]\n")

        # Initialize embedder (lazy loads model on first use)
        embedder = Embedder(settings.embedding_model)

        # Initialize indexer
        indexer = Indexer(source_storage, vector_storage, embedder, settings)

        # Index each source
        total_chunks = 0
        for source in pending:
            try:
                chunks_created = indexer.index_source(source.name)
                total_chunks += chunks_created
                console.print()  # Blank line between sources
            except Exception as e:
                console.print(f"[red]Error indexing '{source.name}': {e}[/red]")
                console.print()

        console.print(f"[green]✓ Indexing complete: {total_chunks} total chunks created[/green]")

    finally:
        source_storage.close()


@app.command()
@handle_errors
def serve():
    """Start the MCP server."""
    settings.ensure_dirs()
    source_storage = SourceStorage(settings.db_path)

    try:
        # Check if there are any indexed sources
        sources = source_storage.list_sources()
        indexed = [s for s in sources if s.indexed_at]

        if not indexed:
            console.print("[red]Error: No sources indexed[/red]")
            console.print("\nPlease add and index sources first:")
            console.print("  1. c42 add <path> --name <name>")
            console.print("  2. c42 index")
            raise typer.Exit(1)

        console.print(f"[green]Starting MCP server with {len(indexed)} indexed source(s)...[/green]")

        # Import and run server
        from .server import run_server
        run_server()

    finally:
        source_storage.close()


if __name__ == "__main__":
    app()
