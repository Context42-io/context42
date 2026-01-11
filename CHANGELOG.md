# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-01-11

### Fixed
- Fixed `c42 status` showing 0 chunks due to undeclared pandas dependency
- Fixed `-v/--verbose` flag not showing errors (silent exception handling)
- Fixed LanceDB deprecation warning (`table_names()` → `list_tables()`)

### Changed
- Rewrote `get_stats()` to use PyArrow instead of pandas (lighter, no extra dependency)
- Added proper `logger.debug()` calls for verbose mode support
- Updated `init_db()` to use `list_tables()` API and access `.tables` attribute

### Removed
- Removed implicit pandas dependency (~12MB smaller installation)

## [0.4.0] - 2026-01-11

### Added
- Exclude patterns with sensible defaults (node_modules, .git, __pycache__, etc)
- `--exclude` flag for custom patterns (can be used multiple times)
- `--no-default-excludes` flag to disable default patterns
- reStructuredText (.rst) file support for Python documentation
- Multi-format indexing (MD + RST in same source)
- RST header parsing with level detection
- RST directive cleaning (code blocks, notes, warnings)
- Exclude patterns stored in database per source

### Changed
- File discovery now uses SUPPORTED_EXTENSIONS config (.md, .rst)
- Indexer reports excluded file count during indexing
- CLI `add` command help text now mentions RST support

## [0.3.3] - 2026-01-10

### Changed
- Refactored indexing to two-phase processing for accurate progress tracking
- Progress bars now show exact counts instead of changing denominators
- Model loading happens before progress bars start

### Fixed
- Fixed batch counter that was incrementing total during processing
- Fixed log spam (49 lines reduced to 2 progress bars)

## [0.3.2] - 2026-01-10

### Added
- Path duplicate prevention in `c42 add`
- Warning when indexing source with all chunks already existing
- `get_source_by_path()` method in storage

### Fixed
- Ghost source creation when adding same path with different name

## [0.3.1] - 2026-01-10

### Added
- Batch processing for indexing (memory optimization)
- `batch_size` configuration option (default: 50)
- Granular progress feedback during indexing

### Changed
- Indexing now processes chunks in batches instead of all at once

## [0.3.0] - 2026-01-10

### Added
- Priority system for sources (0.1-1.0 scale)
- `--priority` flag for `c42 add` command
- `c42 set-priority` command
- `is_priority` field in search results
- Priority-weighted search scoring

### Changed
- Search results now ordered by weighted score (vector_score × priority)
- `c42 list` shows priority column

## [0.2.1] - 2026-01-09

### Fixed
- Critical chunking bug that created header-only chunks
- Chunks now always contain meaningful content beyond headers

## [0.2.0] - 2026-01-09

### Added
- `c42 remove <name>` command to delete sources and their chunks
- `c42 status` command with storage statistics
- `c42 search <query>` CLI command for testing and debugging

## [0.1.0] - 2026-01-09

### Added
- Initial MVP release
- CLI commands: `add`, `list`, `index`, `serve`
- MCP server with `search` tool
- SQLite metadata storage with WAL mode
- LanceDB vector storage
- fastembed integration (BAAI/bge-small-en-v1.5)
- Markdown chunking with sentence boundaries
- Idempotent indexing via content hash
