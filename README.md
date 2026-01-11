# Context42

MCP server for semantic search over your personal coding instructions.

## What it does

Context42 indexes your markdown documentation and makes it searchable by AI assistants like Claude. Your coding standards, guidelines, and patterns become instantly accessible during development.

## Installation

```bash
# Using pipx (recommended)
pipx install context42-io

# Using uv
uvx context42-io

# Using pip
pip install context42-io
```

## Quick Start

```bash
# 1. Add your documentation
c42 add ~/my-coding-standards --name standards

# 2. Index the content
c42 index

# 3. Start the MCP server
c42 serve
```

## Claude Desktop Integration

Add to your Claude Desktop configuration file:

| Platform | Path |
|----------|------|
| Linux | `~/.config/claude/claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

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

Restart Claude Desktop after adding the configuration.

## CLI Commands

| Command | Description |
|---------|-------------|
| `c42 add <path> --name <n>` | Add documentation source |
| `c42 add <path> --priority <0.1-1.0>` | Add with priority (higher = more important) |
| `c42 list` | List all sources |
| `c42 index` | Index pending sources |
| `c42 search <query>` | Search from CLI |
| `c42 status` | Show statistics |
| `c42 remove <n>` | Remove a source |
| `c42 set-priority <n> <value>` | Change source priority |
| `c42 serve` | Start MCP server |

## Priority System

Set higher priority for your personal instructions so they take precedence over reference documentation:

```bash
# Your rules (high priority)
c42 add ~/my-standards --name standards --priority 1.0

# Reference docs (lower priority)
c42 add ~/library-docs --name docs --priority 0.5
```

Search results are weighted by priority, ensuring your preferences appear first.

## Features

- **100% Offline** - Runs locally after initial model download
- **Priority System** - Your rules take precedence over reference docs
- **Fast Indexing** - Progress bars with batch processing
- **Idempotent** - Safe to re-run indexing

## Configuration

Environment variables (optional):

```bash
C42_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"  # Default model
C42_CHUNK_SIZE=500                             # Characters per chunk
C42_BATCH_SIZE=50                              # Chunks per batch
C42_DATA_DIR="~/.local/share/context42"        # Data directory
```

## Advanced Configuration

For users who need faster model downloads or encounter rate limiting:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token for faster downloads and higher rate limits |

```bash
# Optional: Set HuggingFace token for faster downloads
export HF_TOKEN="hf_your_token_here"
```

Get your token at: https://huggingface.co/settings/tokens

## Data Storage

Context42 stores data in platform-specific locations:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/context42/` |
| macOS | `~/Library/Application Support/context42/` |
| Windows | `%LOCALAPPDATA%\context42\` |

## License

MIT
