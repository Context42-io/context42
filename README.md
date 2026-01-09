# Context42

MCP server for personal coding instructions and documentation.

## What is Context42?

Context42 is an MCP (Model Context Protocol) server that provides intelligent context retrieval for LLMs through vector search over your personal coding instructions and documentation. It allows you to index your coding standards, patterns, and guidelines, making them searchable by AI assistants like Claude Desktop.

## Features (MVP v0.1)

- 100% offline operation (after initial model download)
- Vector search over markdown documentation
- Simple CLI for managing sources
- MCP server integration with Claude Desktop, Cursor, and other MCP clients
- Idempotent re-indexing (safe to run multiple times)
- Cross-platform support (Linux, macOS, Windows)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Add a source directory containing markdown files
c42 add ~/my-coding-standards --name standards

# List all sources
c42 list

# Index the sources (creates vector embeddings)
c42 index

# Start the MCP server
c42 serve
```

## MCP Integration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "context42": {
      "command": "c42",
      "args": ["serve"]
    }
  }
}
```

Config file locations:
- **Linux**: `~/.config/claude/claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

## Architecture

- **Embeddings**: fastembed (ONNX, offline, 384-dim bge-small-en-v1.5)
- **Vector DB**: LanceDB (embedded, zero external dependencies)
- **Metadata**: SQLite with WAL mode (atomic writes)
- **Chunking**: Character-based with sentence boundaries (no mid-sentence splits)
- **MCP Framework**: FastMCP

## Storage

Context42 stores data in platform-specific locations:
- **Linux**: `~/.local/share/context42/`
- **macOS**: `~/Library/Application Support/context42/`
- **Windows**: `%LOCALAPPDATA%\context42\`

## Configuration

Configuration via environment variables:

```bash
# Change embedding model
C42_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" c42 index

# Change chunk size (characters)
C42_CHUNK_SIZE=1000 c42 index

# Change data directory
C42_DATA_DIR="/custom/path" c42 index
```

## CLI Commands

- `c42 add <path> --name <name>` - Add a source directory
- `c42 list` - List all sources and their status
- `c42 index` - Index pending sources
- `c42 serve` - Start the MCP server

## MCP Tools

- `search(query: str, top_k: int = 5)` - Search personal instructions using vector similarity

## Current Limitations (MVP v0.1)

- Markdown files only (code chunking with tree-sitter in v0.5)
- Local paths only (Git support in v0.3)
- Vector search only (hybrid search in v0.6)
- Single MCP tool (more tools in v0.2+)

## Development

See `context42-architecture-v2.1.md` for detailed architecture and `context42-implementation-guide.md` for implementation details.

## License

MIT
