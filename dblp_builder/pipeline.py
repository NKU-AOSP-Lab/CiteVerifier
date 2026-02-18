from __future__ import annotations

import gzip
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

ALLOWED_DOWNLOAD_HOSTS = {"dblp.org", "dblp.uni-trier.de"}

PUB_TAGS = {
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "phdthesis",
    "mastersthesis",
    "www",
}

ProgressCallback = Callable[[str, dict[str, Any]], None]
LogCallback = Callable[[str], None]
ShouldStopCallback = Callable[[], bool]


def _validate_download_url(url: str) -> None:
    """Only allow downloads from trusted DBLP hosts to reduce SSRF risk."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    if parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(
            f"Download host not allowed: {parsed.hostname}. "
            f"Only {ALLOWED_DOWNLOAD_HOSTS} are permitted."
        )


@dataclass(slots=True)
class PipelineConfig:
    xml_gz_url: str
    dtd_url: str
    data_dir: Path
    xml_gz_name: str = "dblp.xml.gz"
    xml_name: str = "dblp.xml"
    dtd_name: str = "dblp.dtd"
    db_name: str = "dblp.sqlite"
    batch_size: int = 1000
    progress_every: int = 10000
    rebuild: bool = True

    @property
    def xml_gz_path(self) -> Path:
        return self.data_dir / self.xml_gz_name

    @property
    def xml_path(self) -> Path:
        return self.data_dir / self.xml_name

    @property
    def dtd_path(self) -> Path:
        return self.data_dir / self.dtd_name

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _raise_if_stopped(should_stop: ShouldStopCallback) -> None:
    if should_stop():
        raise InterruptedError("Pipeline stopped by user request.")


def _download_file(
    url: str,
    target_path: Path,
    phase: str,
    log: LogCallback,
    progress: ProgressCallback,
    should_stop: ShouldStopCallback,
) -> None:
    log(f"Downloading {url} -> {target_path}")
    _validate_download_url(url)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        total_raw = response.headers.get("content-length")
        total = int(total_raw) if total_raw and total_raw.isdigit() else None
        downloaded = 0
        last_report = 0

        with target_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                _raise_if_stopped(should_stop)
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 5 * 1024 * 1024:
                    progress(
                        phase,
                        {
                            "downloaded_bytes": downloaded,
                            "total_bytes": total,
                        },
                    )
                    last_report = downloaded

        progress(
            phase,
            {
                "downloaded_bytes": downloaded,
                "total_bytes": total,
            },
        )

    log(f"Download complete: {target_path} ({downloaded} bytes)")


def _decompress_xml(
    source_gz: Path,
    target_xml: Path,
    log: LogCallback,
    progress: ProgressCallback,
    should_stop: ShouldStopCallback,
) -> None:
    log(f"Decompressing {source_gz} -> {target_xml}")
    target_xml.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    last_report = 0
    with gzip.open(source_gz, "rb") as src, target_xml.open("wb") as dst:
        while True:
            _raise_if_stopped(should_stop)
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)
            if written - last_report >= 20 * 1024 * 1024:
                progress("decompress_xml", {"written_bytes": written})
                last_report = written

    progress("decompress_xml", {"written_bytes": written})
    log(f"Decompression complete: {target_xml} ({written} bytes)")


def _init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            year INTEGER,
            venue TEXT,
            pub_type TEXT,
            raw_xml TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pub_authors (
            pub_id INTEGER NOT NULL,
            author_id INTEGER NOT NULL,
            FOREIGN KEY(pub_id) REFERENCES publications(id),
            FOREIGN KEY(author_id) REFERENCES authors(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pub_authors_pub ON pub_authors(pub_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pub_authors_author ON pub_authors(author_id);")

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS title_fts
        USING fts5(title, content='publications', content_rowid='id');
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS author_fts
        USING fts5(name, content='authors', content_rowid='id');
        """
    )
    conn.commit()


def _extract_year_venue(elem: Any) -> tuple[int | None, str | None]:
    year = None
    venue = None

    year_elem = elem.find("year")
    if year_elem is not None and year_elem.text:
        try:
            year = int(_normalize(year_elem.text))
        except ValueError:
            year = None

    journal_elem = elem.find("journal")
    booktitle_elem = elem.find("booktitle")
    if journal_elem is not None and journal_elem.text:
        venue = _normalize(journal_elem.text)
    elif booktitle_elem is not None and booktitle_elem.text:
        venue = _normalize(booktitle_elem.text)

    return year, venue


def _build_db(
    xml_path: Path,
    db_path: Path,
    batch_size: int,
    progress_every: int,
    log: LogCallback,
    progress: ProgressCallback,
    should_stop: ShouldStopCallback,
) -> dict[str, Any]:
    try:
        from lxml import etree as ET
    except Exception as exc:
        raise RuntimeError(
            "lxml is required for building from dblp.xml. Install it in the runtime environment."
        ) from exc

    log(f"Building sqlite db from {xml_path} -> {db_path}")
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    cur = conn.cursor()

    insert_pub = (
        "INSERT INTO publications(title, year, venue, pub_type, raw_xml) "
        "VALUES (?, ?, ?, ?, ?);"
    )

    insert_author = "INSERT OR IGNORE INTO authors(name) VALUES (?);"
    insert_pub_author = "INSERT INTO pub_authors(pub_id, author_id) VALUES (?, ?);"
    insert_title_fts = "INSERT INTO title_fts(rowid, title) VALUES (?, ?);"
    insert_author_fts = "INSERT INTO author_fts(rowid, name) VALUES (?, ?);"

    author_cache: dict[str, int] = {}
    pending_pub_authors: list[tuple[int, int]] = []
    pending_titles: list[tuple[int, str]] = []
    pending_authors: list[tuple[int, str]] = []

    # Secure XML parser: allow local DTD for legitimate character entities
    # (e.g. &auml;) but block external SYSTEM entity resolution to prevent XXE.
    class _SafeResolver(ET.Resolver):
        def resolve(self, system_url, public_id, context):
            if system_url and system_url.endswith(".dtd"):
                return self.resolve_filename(system_url, context)
            return self.resolve_string("", context)

    parser = ET.XMLParser(
        load_dtd=True,
        resolve_entities=True,
        huge_tree=True,
        no_network=True,
    )
    parser.resolvers.add(_SafeResolver())

    context = ET.iterparse(
        str(xml_path),
        events=("end",),
        parser=parser,
    )

    count = 0
    start = time.time()
    last_report = start
    try:
        for _, elem in context:
            _raise_if_stopped(should_stop)
            if elem.tag not in PUB_TAGS:
                continue

            title_elem = elem.find("title")
            if title_elem is None or title_elem.text is None:
                elem.clear()
                continue

            title = _normalize(title_elem.text)
            year, venue = _extract_year_venue(elem)
            raw_xml = ET.tostring(elem, encoding="unicode")
            cur.execute(insert_pub, (title, year, venue, elem.tag, raw_xml))

            pub_id = cur.lastrowid
            pending_titles.append((pub_id, title))

            for author_elem in elem.findall("author"):
                if author_elem.text is None:
                    continue
                author = _normalize(author_elem.text)
                author_id = author_cache.get(author)
                if author_id is None:
                    cur.execute(insert_author, (author,))
                    cur.execute("SELECT id FROM authors WHERE name = ?;", (author,))
                    row = cur.fetchone()
                    if row is None:
                        continue
                    author_id = row[0]
                    author_cache[author] = author_id
                    pending_authors.append((author_id, author))
                pending_pub_authors.append((pub_id, author_id))

            count += 1
            if count % batch_size == 0:
                cur.executemany(insert_pub_author, pending_pub_authors)
                cur.executemany(insert_title_fts, pending_titles)
                cur.executemany(insert_author_fts, pending_authors)
                pending_pub_authors.clear()
                pending_titles.clear()
                pending_authors.clear()
                conn.commit()

            if progress_every > 0 and count % progress_every == 0:
                now = time.time()
                elapsed = max(now - start, 0.001)
                interval = now - last_report
                if interval >= 0.5:
                    progress(
                        "build_db",
                        {
                            "processed_records": count,
                            "records_per_sec": round(count / elapsed, 2),
                        },
                    )
                    last_report = now

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        if pending_pub_authors:
            cur.executemany(insert_pub_author, pending_pub_authors)
        if pending_titles:
            cur.executemany(insert_title_fts, pending_titles)
        if pending_authors:
            cur.executemany(insert_author_fts, pending_authors)

        conn.commit()
    finally:
        conn.close()

    elapsed = max(time.time() - start, 0.001)
    rate = round(count / elapsed, 2)
    progress("build_db", {"processed_records": count, "records_per_sec": rate})
    log(f"Build complete: {count} records, {rate} rec/s")
    return {
        "processed_records": count,
        "elapsed_seconds": round(elapsed, 2),
        "records_per_sec": rate,
        "db_path": str(db_path),
    }


def _cleanup_db_files(db_path: Path, log: LogCallback) -> None:
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink(missing_ok=True)
            log(f"Removed existing file: {path}")


def run_pipeline(
    config: PipelineConfig,
    log: LogCallback,
    progress: ProgressCallback,
    should_stop: ShouldStopCallback,
) -> dict[str, Any]:
    started = time.time()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    log(f"Pipeline start (data_dir={config.data_dir})")

    if config.rebuild:
        _cleanup_db_files(config.db_path, log)

    _raise_if_stopped(should_stop)
    _download_file(
        config.dtd_url,
        config.dtd_path,
        "download_dtd",
        log,
        progress,
        should_stop,
    )

    _raise_if_stopped(should_stop)
    _download_file(
        config.xml_gz_url,
        config.xml_gz_path,
        "download_xml_gz",
        log,
        progress,
        should_stop,
    )

    _raise_if_stopped(should_stop)
    _decompress_xml(
        config.xml_gz_path,
        config.xml_path,
        log,
        progress,
        should_stop,
    )

    _raise_if_stopped(should_stop)
    build_stats = _build_db(
        xml_path=config.xml_path,
        db_path=config.db_path,
        batch_size=config.batch_size,
        progress_every=config.progress_every,
        log=log,
        progress=progress,
        should_stop=should_stop,
    )

    elapsed = round(time.time() - started, 2)
    result = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "xml_gz_path": str(config.xml_gz_path),
        "xml_path": str(config.xml_path),
        "dtd_path": str(config.dtd_path),
        **build_stats,
    }
    log(f"Pipeline finished in {elapsed}s")
    return result
