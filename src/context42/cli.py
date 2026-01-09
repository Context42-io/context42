"""CLI commands for Context42."""

from pathlib import Path
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


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Context42 - MCP server for personal coding instructions."""
    if ctx.invoked_subcommand is None:
        console.print("[yellow]Context42 - MCP server for personal coding instructions[/yellow]")
        console.print("\nUse --help for available commands")
        console.print("\nQuick start:")
        console.print("  c42 add <path> --name <name>  # Add source directory")
        console.print("  c42 list                      # List sources")
        console.print("  c42 index                     # Index sources")
        console.print("  c42 serve                     # Start MCP server")


@app.command()
def add(
    path: str = typer.Argument(..., help="Path to source directory"),
    name: str = typer.Option(..., "--name", "-n", help="Unique source name")
):
    """Add a source directory containing markdown files."""
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

    # Add source
    try:
        storage.add_source(name, str(source_path))
        console.print(f"[green]✓ Source '{name}' added successfully[/green]")
        console.print(f"[blue]Path: {source_path}[/blue]")
        console.print("\n[yellow]Next: Run 'c42 index' to index this source[/yellow]")
    except sqlite3.IntegrityError:
        console.print(f"[red]Error: Source '{name}' already exists[/red]")
        raise typer.Exit(1)
    finally:
        storage.close()


@app.command()
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
        table.add_column("Indexed", style="green")
        table.add_column("Model", style="magenta")

        for source in sources:
            indexed_status = source.indexed_at if source.indexed_at else "[yellow]pending[/yellow]"
            model_display = source.embedding_model if source.embedding_model else "-"

            table.add_row(
                source.name,
                source.path,
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
