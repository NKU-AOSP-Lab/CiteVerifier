# CiteVerifier: Citation Verification System

This directory contains the implementation of CiteVerifier, an end-to-end citation verification system that parses bibliographic references, queries multiple authoritative sources, and classifies citation validity.

### Core System

#### `verifier.py` (Main Entry Point)
High-level interface for citation verification. Orchestrates the parsing, matching, and classification pipeline using a multi-strategy approach:
1. **DBLP Pre-match (Local)**: Match the title against a local DBLP title database (ratio-based)
2. **Local Cache**: Query SQLite database for previously verified references
3. **API Search**: Google Scholar API via ScrapingDog for validation
4. **Google Fallback**: Fallback to standard Google search if Scholar API fails
5. **LLM Reparse**: Use LLM to re-extract metadata from malformed citations, then retry

**Usage:**
```bash
python verifier.py --input references.json --output verified_results.json
python verifier.py --dir batch_dir/ --database scholar_results.db
python verifier.py --sample  # Test with built-in examples
```

#### `parser/` (Reference Parsing)
Extracts and normalizes bibliographic metadata from different citation formats.

**Key modules:**
- `parser/bibparser.py` - BibTeX parsing and normalization
- `parser/reference_parser.py` - Plain text/formatted citation parsing
- `parser/metadata_extractor.py` - Field extraction (authors, title, year, venue)
- `parser/llm_parser.py` - LLM-based metadata reparse for malformed citations

#### `checker/` (Validation & Matching)
Queries multiple external databases and performs similarity-based matching.

**Key modules:**
- `checker/dblp_checker.py` - Query against DBLP Computer Science Bibliography
- `checker/scholar_checker.py` - Google Scholar search via ScrapingDog API
- `checker/google_checker.py` - Fallback Google search when API unavailable
- `checker/similarity_matcher.py` - String and semantic similarity matching

### Supporting Utilities

#### `reference_storage_service.py`
Local SQLite caching and storage of reference resolution results to minimize external API calls.

#### `parsed_references_database.py`
Database schema and utilities for storing parsed reference metadata with indexed queries.

#### `unified_database.py`
Unified interface to multiple bibliographic data sources.

#### `grobid_parser_to_xml.py`
Integration with GROBID for extracting reference sections from PDF documents.

```python
from grobid_parser_to_xml import grobid_parse
refs = grobid_parse("path/to/pdf_dir", "output/xml_dir")
```

## Verification Pipeline

1. **Input**: Citation string(s) in various formats (BibTeX, plain text, JSON)
2. **Parse**: Extract metadata (authors, title, year, venue)
3. **Preprocess**: Filter invalid entries; reparse using LLM if title is empty
4. **Verify Chain**:
  - DBLP pre-match using local DB (ratio-based title match)
  - Check local cache (SQLite database)
  - Query Google Scholar API
  - Fallback to Google search if needed
  - Retry with LLM-reparsed metadata if all else fails
5. **Output**: CSV with verification results and classified validity

**Matching policy:** Title matching uses ratio-based similarity across DBLP pre-match and online sources to keep matching behavior consistent.

## Setup

### Installation
```bash
pip install pandas tqdm grobid-client-python aiohttp reportlab
```

### DBLP Pre-match (Recommended)
The verifier runs a DBLP title pre-match before online lookups. Build the local DBLP title database once:

```bash
# Place dblp.xml and dblp.dtd in the same folder as dblp_match.py
python dblp_match.py --build-db --dblp-xml dblp.xml --db dblp_titles.db
```

Then run CiteVerifier with the DB path (default: `dblp_titles.db`).

### Configuration
API keys must be configured in `checker/config.py` or environment:
- **ScrapingDog API Key**: For Google Scholar access
- **Google Search API Key**: Optional, for explicit fallback search

Set via:
```bash
export SCRAPINGDOG_KEY="your_key"
export GOOGLE_SEARCH_KEY="your_key"
```

## Command Line Reference

### Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--input` | `-i` | Single JSON file to verify | None |
| `--dir` | `-d` | Directory of JSON files to batch process | None |
| `--inllm` | `-l` | LLM output directory (special format handling) | None |
| `--output` | `-o` | Output JSON file path | `scholar_results.json` |
| `--database` | `-db` | SQLite database path | `scholar_results.db` |
| `--concurrent` | `-c` | Concurrent requests | 10 |
| `--sample` | | Use built-in test data | False |
| `--stats-only` | | Display database statistics only | False |
| `--clear-db` | | Clear database contents | False |
| `--cleanup-duplicates` | | Remove duplicate entries from database | False |
| `--dblp-db` | | DBLP SQLite database path | `dblp_titles.db` |
| `--dblp-threshold` | | DBLP title match threshold (ratio) | `0.9` |
| `--dblp-max-candidates` | | Max DBLP candidates per query | `100000` |
| `--disable-dblp` | | Skip DBLP pre-match step | False |

### Usage Examples

```bash
# Verify single file
python verifier.py --input references.json --output results.json

# Batch process directory
python verifier.py --dir ./batch_refs/ --concurrent 20

# Process LLM output
python verifier.py --inllm ./llm_generated/ --database verified.db

# View statistics
python verifier.py --stats-only --database verified.db

# Test with examples
python verifier.py --sample
```

## Output Format

**Main output CSV** (`*_verified.csv`):
- `reference_id`: Unique reference identifier
- `final_status`: Classification (VALID, INVALID, SUSPICIOUS, UNVERIFIED)
- `diagnosis`: Summary of findings  
- `title_similarity`: Title match score (0.0-1.0)
- `original_title`: Submitted citation title
- `found_title`: Database match title
- `sources_checked`: Methods used (cache, scrapingdog, google_search, llm_reparse)
- `verification_notes`: Detailed notes from verification process
- `match_confidence`: Confidence score (0.0-1.0)

**Database** (`scholar_results.db`):
SQLite database with tables for cached results, enabling faster subsequent runs and statistical analysis.

## Reproduce Main Results

```bash
# Verify archival papers (from Archival Analysis section)
python verifier.py --dir ../Archival\ Analysis/ --output archival_verified.csv

# Verify LLM-generated citations (from LLM Benchmark section)
python verifier.py --inllm ../LLM\ Benchmark/LLM-Generated\ References/ \
  --output llm_verification_results.csv

# Export comprehensive statistics
python verifier.py --stats-only --database scholar_results.db > verification_stats.txt
```

## Performance Notes

- **Caching**: First run is slower; subsequent runs use cached results
- **Rate Limiting**: Avoid high concurrency (>20) to prevent Scholar API blocking  
- **Batch Size**: For large batches (>1000 refs), split into smaller jobs
- **Timeout**: Each API call has 30-second timeout; adjust in `checker/config.py` if needed

## Troubleshooting

- **"API rate limit exceeded"**: Reduce `--concurrent` parameter; retry later
- **"Database locked"**: Ensure no other processes access the database
- **"Invalid reference format"**: Tool attempts LLM reparse; check input format
- **"Empty results"**: Verify API keys are set; check internet connectivity

## Dependencies

```
pandas>=1.3.0
tqdm>=4.60.0
aiohttp>=3.8.0
reportlab>=3.6.0
grobid-client-python>=0.0.1.dev5
requests>=2.26.0
```

See `requirements.txt` for exact versions.

## Files

- `verifier.py` - Main entry point
- `parser/` - Citation parsing modules
- `checker/` - Verification backend modules
- `dblp_match.py` - Build/search local DBLP title database for pre-match
- `requirements.txt` - Python dependencies
- `config.py` - Configuration parameters

## Contact & Questions

For implementation details, troubleshooting, or methodology questions, see the main paper's "Methodology" section (Section 3) and "Archival Results" (Section 6).

---