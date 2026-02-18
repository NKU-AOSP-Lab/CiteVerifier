import argparse
import os
import html
import multiprocessing as mp
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from lxml import etree
from rapidfuzz import fuzz, process

# Globals in worker process (set by pool initializer)
_worker_all_titles: Optional[List[Tuple[int, str, str]]] = None
_worker_conn: Optional[sqlite3.Connection] = None
_worker_max_candidates: int = 100000


def _fix_bare_ampersands(s: str) -> str:
    """
    Replace bare & that are not the start of a valid XML/HTML entity with &amp;.
    Valid: &name; (alphanumeric) or &#123; or &#x1a2b;. Also fixes empty ref "&;".
    """
    return re.sub(r"&(?!(?:[a-zA-Z0-9]+|#\d+|#x[0-9a-fA-F]+);)", "&amp;", s)


def _fix_ampersands_aggressive(s: str) -> str:
    """
    Ultra-aggressive entity fixing: convert ALL entities to their Unicode equivalents,
    and escape any remaining & that could cause parsing issues.
    """
    # First, try to decode all HTML entities to Unicode
    s = html.unescape(s)
    
    # Now escape any remaining bare & that aren't part of valid XML entities
    # Valid XML entities: &amp; &lt; &gt; &quot; &apos; and numeric &#...; &#x...;
    # We'll escape ALL & and then restore only the 5 XML built-in entities
    s = s.replace("&", "&amp;")
    
    # Restore the 5 XML built-in entities (they should remain as entities)
    s = s.replace("&amp;amp;", "&amp;")
    s = s.replace("&amp;lt;", "&lt;")
    s = s.replace("&amp;gt;", "&gt;")
    s = s.replace("&amp;quot;", "&quot;")
    s = s.replace("&amp;apos;", "&apos;")
    
    return s


class _EntityReplacingReader:
    """
    File-like object that streams a text file and replaces HTML/XML entities
    (e.g. &uuml;, &auml;) so that lxml can parse dblp.xml without DTD.
    Reads line-by-line to avoid splitting entities across chunks.
    Exposes .bytes_read for progress (bytes consumed from the file so far).
    aggressive=True: replace every & then decode entities (avoids "entity ref: no name").
    """

    def __init__(self, path: Path, aggressive: bool = True) -> None:
        self._path = path
        self._aggressive = aggressive
        self._buffer = b""
        self._iter: Optional[Iterator[bytes]] = None
        self.bytes_read: int = 0

    def _line_stream(self) -> Iterator[bytes]:
        with open(self._path, "rb") as f:
            for line in f:
                self.bytes_read += len(line)
                text = line.decode("utf-8", errors="replace")
                if self._aggressive:
                    text = _fix_ampersands_aggressive(text)
                else:
                    text = _fix_bare_ampersands(text)
                    text = html.unescape(text)
                yield text.encode("utf-8")

    def read(self, size: int = -1) -> bytes:
        if self._iter is None:
            self._iter = self._line_stream()
        if size == -1:
            if self._buffer:
                result = self._buffer
                self._buffer = b""
                try:
                    result += b"".join(self._iter)
                except StopIteration:
                    pass
                return result
            try:
                return b"".join(self._iter)
            except StopIteration:
                return b""
        while len(self._buffer) < size:
            try:
                self._buffer += next(self._iter)
            except StopIteration:
                break
        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return result


def normalize_title(s: str) -> str:
    """Simple normalization for titles."""
    return " ".join(str(s).lower().split())


# ============== SQLite Database Functions ==============

def build_dblp_db(
    dblp_xml: Path,
    db_path: Path,
    batch_size: int = 50000,
) -> None:
    """
    Deprecated.

    DB building is no longer provided by CiteVerifier. Use the DblpService backend to build
    `dblp.sqlite` in fullmeta mode, then point `verifier.py --dblp-db` to that database.
    """
    raise RuntimeError(
        "Building DBLP databases from CiteVerifier is disabled. "
        "Use DblpService to build dblp.sqlite, then pass --dblp-db to verifier.py."
    )


def _sqlite_readonly_fast(conn: sqlite3.Connection) -> None:
    """Tune SQLite for read-heavy workload (faster, less safe for writes)."""
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA cache_size = -256000")  # 256MB
    conn.execute("PRAGMA temp_store = MEMORY")


def _db_has_title_fts(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='title_fts'")
    return cur.fetchone() is not None


def _fts_query_from_norm(title_norm: str, max_terms: int = 5) -> str:
    # Keep only alnum tokens to avoid FTS query syntax pitfalls.
    # (isalnum keeps Unicode letters/digits; we still rely on DB tokenizer for actual matching.)
    cleaned = "".join((ch.lower() if ch.isalnum() else " ") for ch in title_norm)
    tokens = cleaned.split()
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= max_terms:
            break
    return " ".join(uniq)


def _search_dblp_by_title_fts(
    conn: sqlite3.Connection,
    orig_title: str,
    max_candidates: int,
) -> Optional[Dict[str, Any]]:
    title_norm = normalize_title(orig_title)
    query = _fts_query_from_norm(title_norm)
    if not query:
        return None

    limit = int(max_candidates)
    # Defensive cap: very broad queries can be expensive.
    if limit <= 0:
        limit = 200
    limit = min(limit, 10000)

    cur = conn.cursor()
    cur.execute(
        "SELECT rowid, title FROM title_fts WHERE title_fts MATCH ? LIMIT ?;",
        (query, limit),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    best_id = None
    best_title = None
    best_score = -1.0

    for rid, title in rows:
        cand_norm = normalize_title(title)
        score = fuzz.token_sort_ratio(title_norm, cand_norm) / 100.0
        if score > best_score:
            best_score = score
            best_id = rid
            best_title = title

    if best_id is None or best_title is None:
        return None

    return {
        "dblp_id": int(best_id),
        "dblp_title": str(best_title),
        "dblp_title_similarity": float(best_score),
    }


def search_dblp_by_index(
    conn: sqlite3.Connection,
    orig_title: Any,
    max_candidates: int = 100000,
) -> Optional[Dict[str, Any]]:
    """Fuzzy match by normalized title using ratio (edit-distance based)."""
    if not isinstance(orig_title, str) or not orig_title.strip():
        return None

    if _db_has_title_fts(conn):
        return _search_dblp_by_title_fts(conn, orig_title, max_candidates=max_candidates)

    title_norm = normalize_title(orig_title)
    words = [w for w in title_norm.split() if len(w) >= 2]
    if not words:
        return None
    if len(words) < 2:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, title_norm FROM titles WHERE title_norm = ? LIMIT 1",
            (title_norm,),
        )
        row = cur.fetchone()
        if not row:
            return None
        best_id, best_title, _ = row
        return {
            "dblp_id": best_id,
            "dblp_title": best_title,
            "dblp_title_similarity": 1.0,
        }

    cur = conn.cursor()
    placeholders = ",".join("?" * len(words))
    cur.execute(
        f"""
        SELECT t.id, t.title, t.title_norm
        FROM titles t
        INNER JOIN (
            SELECT id FROM title_words
            WHERE word IN ({placeholders})
            GROUP BY id
            ORDER BY COUNT(*) DESC
            LIMIT ?
        ) c ON t.id = c.id
        """,
        (*words, max_candidates),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    titles_norm_list = [r[2] for r in rows]
    _, score, idx = process.extractOne(
        title_norm,
        titles_norm_list,
        scorer=fuzz.ratio,
    )
    best_id, best_title, _ = rows[idx]
    return {
        "dblp_id": best_id,
        "dblp_title": best_title,
        "dblp_title_similarity": score / 100.0,
    }


def _db_has_word_index(conn: sqlite3.Connection) -> bool:
    # "Indexed" here means we can do candidate selection without loading all titles.
    if _db_has_title_fts(conn):
        return True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='title_words'")
    return cur.fetchone() is not None


def load_all_titles_from_db(db_path: Path, quiet: bool = False) -> List[Tuple[int, str, str]]:
    """Load all (id, title, title_norm) from database into memory for brute-force search."""
    if not quiet:
        print(f"[load_all_titles_from_db] Loading from {db_path}...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='titles'")
    has_titles = cur.fetchone() is not None
    if has_titles:
        cur.execute("SELECT id, title, title_norm FROM titles")
        rows = cur.fetchall()
    else:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='publications'")
        has_pubs = cur.fetchone() is not None
        if not has_pubs:
            conn.close()
            raise RuntimeError(f"Unsupported DB schema in {db_path}")
        cur.execute("SELECT id, title FROM publications")
        pub_rows = cur.fetchall()
        rows = [(int(pid), str(title), normalize_title(title)) for (pid, title) in pub_rows]
    conn.close()
    if not quiet:
        print(f"[load_all_titles_from_db] Loaded {len(rows)} titles.")
    return rows


def search_dblp_brute_force(
    all_titles: List[Tuple[int, str, str]],  # (id, title, title_norm)
    orig_title: Any,
) -> Optional[Dict[str, Any]]:
    """Fuzzy match by normalized title using ratio (edit-distance based)."""
    if not isinstance(orig_title, str) or not orig_title.strip():
        return None

    title_norm = normalize_title(orig_title)
    words = [w for w in title_norm.split() if len(w) >= 2]
    if not words:
        return None
    if len(words) < 2:
        for tid, title, norm in all_titles:
            if norm == title_norm:
                return {
                    "dblp_id": tid,
                    "dblp_title": title,
                    "dblp_title_similarity": 1.0,
                }
        return None

    titles_norm_list = [t[2] for t in all_titles]
    _, score, idx = process.extractOne(
        title_norm,
        titles_norm_list,
        scorer=fuzz.ratio,
    )
    best_id, best_title, _ = all_titles[idx]
    return {
        "dblp_id": best_id,
        "dblp_title": best_title,
        "dblp_title_similarity": score / 100.0,
    }


def _match_worker_init(db_path: Path) -> None:
    """Load full DB into worker (high memory). Used only when no word index."""
    global _worker_all_titles
    _worker_all_titles = load_all_titles_from_db(db_path, quiet=True)


def _match_one_title(orig_title: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Worker: brute-force over full titles (uses _worker_all_titles)."""
    global _worker_all_titles
    if _worker_all_titles is None:
        return (orig_title, None)
    result = search_dblp_brute_force(_worker_all_titles, orig_title)
    return (orig_title, result)


def _match_worker_init_index(args: Tuple[Path, int]) -> None:
    """Open DB connection in worker (low memory). No full load."""
    global _worker_conn, _worker_max_candidates
    db_path, max_cand = args
    _worker_conn = sqlite3.connect(str(db_path))
    _sqlite_readonly_fast(_worker_conn)
    _worker_max_candidates = max_cand


def _match_one_title_index(orig_title: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Worker: index-based search (uses _worker_conn, loads only candidates)."""
    global _worker_conn, _worker_max_candidates
    if _worker_conn is None:
        return (orig_title, None)
    result = search_dblp_by_index(_worker_conn, orig_title, _worker_max_candidates)
    return (orig_title, result)


def run_matching(
    v2_csv: Path,
    out_csv: Path,
    db_path: Path,
    dblp_match_threshold: float = 0.9,
    match_all: bool = False,
    title_sim_threshold: float = 0.9,
    sim_column: str = "title_similarity",
    workers: int = 0,
    max_candidates: int = 100000,
    checkpoint_interval: int = 0,
    resume: bool = False,
) -> None:
    """
    Search DBLP by original_title and attach best match + dblp similarity.
    When not match_all, filter by sim_column < title_sim_threshold (e.g. similarity_correct < 0.9).
    If checkpoint_interval > 0, save partial results every N unique titles.
    If resume and checkpoint exists, skip already-done titles and continue.
    """
    print(f"[run_matching] v2_csv = {v2_csv}")
    print(f"[run_matching] out_csv = {out_csv}")
    print(f"[run_matching] db_path = {db_path}")
    checkpoint_path = out_csv.parent / (out_csv.stem + "_checkpoint.csv")

    # Load df: from checkpoint if resume and checkpoint exists, else from v2_csv
    if resume and checkpoint_path.exists():
        print(f"[run_matching] Resuming from checkpoint: {checkpoint_path}")
        df = pd.read_csv(checkpoint_path)
    else:
        df = pd.read_csv(v2_csv)

    if "original_title" not in df.columns:
        raise ValueError("Input CSV must contain an 'original_title' column.")

    # Ensure dblp columns exist (for checkpoint / resume)
    for col in ("dblp_id", "dblp_title", "dblp_title_similarity"):
        if col not in df.columns:
            df[col] = None

    if match_all:
        target_indices = df.index.tolist()
        print(f"[run_matching] match_all=True: verifying all {len(target_indices)} rows with DBLP.")
    else:
        if sim_column not in df.columns:
            raise ValueError(f"Input CSV must contain column '{sim_column}' when not using --match-all.")
        sim_vals = pd.to_numeric(df[sim_column], errors="coerce")
        mask_low = sim_vals < title_sim_threshold
        target_indices = df.index[mask_low].tolist()
        print(f"[run_matching] Total rows: {len(df)}, to verify ({sim_column}<{title_sim_threshold}): {len(target_indices)}")

    # Dedupe only for search: same original_title searched once, result applied to ALL rows with that title
    title_to_indices: Dict[str, List[int]] = {}
    for idx in target_indices:
        orig_title = str(df.loc[idx, "original_title"]).strip()
        title_to_indices.setdefault(orig_title, []).append(idx)
    unique_titles = list(title_to_indices.keys())
    print(f"[run_matching] Unique titles to search: {len(unique_titles)} (result will fill all {len(target_indices)} target rows; output keeps full {len(df)} rows).")

    # Resume: skip titles that already have dblp_title_similarity set
    if resume and checkpoint_path.exists():
        done_titles = set()
        for t in unique_titles:
            idx0 = title_to_indices[t][0]
            val = df.loc[idx0, "dblp_title_similarity"]
            if pd.notna(val) and str(val).strip() != "":
                done_titles.add(t)
        unique_titles = [t for t in unique_titles if t not in done_titles]
        print(f"[run_matching] Resuming: {len(done_titles)} already done, {len(unique_titles)} remaining.")

    if not unique_titles:
        print("[run_matching] Nothing to do (all done or empty). Saving full table.")
        df["dblp_match_status"] = df["dblp_title_similarity"].apply(
            lambda s: "dblp_match" if (pd.notna(s) and float(s) >= dblp_match_threshold) else "dblp_unmatched"
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[run_matching] Saved {out_csv} ({len(df)} rows).")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return

    conn = sqlite3.connect(str(db_path))
    use_index = _db_has_word_index(conn)
    conn.close()

    if use_index:
        print("[run_matching] Using word index (low memory).")
    else:
        print("[run_matching] No index found. Falling back to full-DB load (high memory).")
        workers = 1

    if workers <= 0:
        workers = max(1, mp.cpu_count() - 1) if use_index else 1

    t0 = time.perf_counter()
    empty_result = {"dblp_id": None, "dblp_title": None, "dblp_title_similarity": 0.0}
    save_interval = checkpoint_interval if checkpoint_interval > 0 else max(len(unique_titles), 1)

    # Full-DB load once when not using index (used for all chunks)
    all_titles_cached: Optional[List[Tuple[int, str, str]]] = None
    if not use_index:
        all_titles_cached = load_all_titles_from_db(db_path)

    def _safe_write_csv(path: Path, frame: pd.DataFrame) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            frame.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
        except PermissionError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def apply_results_and_maybe_checkpoint(
        title_to_result: Dict[str, Optional[Dict[str, Any]]],
        chunk_label: str,
    ) -> None:
        # Apply match result to ALL rows that share this title (no rows dropped)
        for orig_title, result in title_to_result.items():
            match_info = result if result else empty_result
            for idx in title_to_indices[orig_title]:
                for k, v in match_info.items():
                    df.loc[idx, k] = v
        if checkpoint_interval > 0:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_csv(checkpoint_path, df)
            print(f"[run_matching] Checkpoint saved ({chunk_label}), full table {len(df)} rows.", flush=True)

    # Process in chunks (one big chunk if checkpoint_interval=0)
    offset = 0
    while offset < len(unique_titles):
        chunk_titles = unique_titles[offset : offset + save_interval]
        offset += len(chunk_titles)

        title_to_result_chunk: Dict[str, Optional[Dict[str, Any]]] = {}
        if use_index:
            if workers <= 1:
                conn = sqlite3.connect(str(db_path))
                _sqlite_readonly_fast(conn)
                for orig_title in chunk_titles:
                    title_to_result_chunk[orig_title] = search_dblp_by_index(conn, orig_title, max_candidates)
                conn.close()
            else:
                csize = max(1, len(chunk_titles) // workers)
                with mp.Pool(workers, initializer=_match_worker_init_index, initargs=((db_path, max_candidates),)) as pool:
                    results = pool.map(_match_one_title_index, chunk_titles, chunksize=csize)
                title_to_result_chunk = {orig: res for orig, res in results}
        else:
            assert all_titles_cached is not None
            for orig_title in chunk_titles:
                title_to_result_chunk[orig_title] = search_dblp_brute_force(all_titles_cached, orig_title)

        apply_results_and_maybe_checkpoint(
            title_to_result_chunk,
            f"{offset}/{len(unique_titles)} unique titles",
        )
        if offset < len(unique_titles):
            print(f"[run_matching] Progress {offset}/{len(unique_titles)} unique titles...", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[run_matching] Done in {elapsed:.1f} s (~{len(target_indices) / max(elapsed, 0.001):.0f} rows/s).")

    def classify(s: Any) -> str:
        try:
            return "dblp_match" if float(s) >= dblp_match_threshold else "dblp_unmatched"
        except Exception:
            return "dblp_unmatched"

    df["dblp_match_status"] = df["dblp_title_similarity"].apply(classify)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_csv(out_csv, df)
    print(f"[run_matching] Saved full results to {out_csv} ({len(df)} rows, same as input).")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[run_matching] Checkpoint removed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match CSV rows against a local DBLP sqlite database (dblp.sqlite). "
            "Build the database via DblpService."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("dblp.sqlite"),
        help="Path to SQLite database (default: ./dblp.sqlite)",
    )
    
    # Matching options
    parser.add_argument(
        "--v2-csv",
        type=Path,
        default=Path("llm_new_res_v2.csv"),
        help="Input CSV file (default: ./llm_new_res_v2.csv)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("llm_new_res_v3.csv"),
        help="Output CSV file (default: ./llm_new_res_v3.csv)",
    )
    parser.add_argument(
        "--title-sim-threshold",
        type=float,
        default=0.9,
        help="Only verify records with sim-column < this value (default: 0.9).",
    )
    parser.add_argument(
        "--sim-column",
        type=str,
        default="similarity_correct",
        help="Column name for filtering (e.g. similarity_correct). Used with title-sim-threshold (default: title_similarity).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N unique titles (0 = disabled). Use with --resume for checkpoint resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip already-verified titles).",
    )
    parser.add_argument(
        "--dblp-match-threshold",
        type=float,
        default=0.9,
        help="Threshold for considering a dblp match as successful (default: 0.9).",
    )
    parser.add_argument(
        "--match-all",
        action="store_true",
        help="Verify every row with DBLP (by original_title).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = CPU count - 1). Use 1 to disable parallel.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="Max candidate titles per query (default: 5000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        print(f"[ERROR] Database not found: {args.db}")
        print("Build dblp.sqlite via DblpService, then re-run with --db.")
        return

    run_matching(
        v2_csv=args.v2_csv,
        out_csv=args.out_csv,
        db_path=args.db,
        dblp_match_threshold=args.dblp_match_threshold,
        match_all=args.match_all,
        title_sim_threshold=args.title_sim_threshold,
        sim_column=args.sim_column,
        workers=args.workers,
        max_candidates=args.max_candidates,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

