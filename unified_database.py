"""
Unified database - Integrates scholar search and general search results into a single database
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker.models import ExternalReference

logger = logging.getLogger(__name__)


@dataclass
class ScholarRecord:
    """Scholar search record data structure"""
    id: Optional[int] = None
    
    # Basic information
    title: str = None
    authors: str = None  # Author list, separated by commas
    year: Optional[int] = None
    venue: str = None
    url: str = None

    # Timestamp
    created_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'url': self.url,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class SearchResultRecord:
    """Search result record data structure"""
    id: Optional[int] = None
    
    # Basic information
    title: str = None
    authors: str = None  # Author list, separated by commas
    venue: str = None
    year: Optional[int] = None
    url: str = None
    
    # Search-related information
    source: str = None  # Search source: google_search, scrapingdog
    search_query: str = None  # Search query
    search_engine: str = None  # Search engine: google, google_scholar
    
    # Metadata
    metadata: str = None  # Additional metadata in JSON format
    result_position: Optional[int] = None  # Position in search results
    
    # Timestamp
    created_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'venue': self.venue,
            'year': self.year,
            'url': self.url,
            'source': self.source,
            'search_query': self.search_query,
            'search_engine': self.search_engine,
            'metadata': self.metadata,
            'result_position': self.result_position,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UnifiedDatabase:
    """Unified database - Contains scholar_results and search_results tables"""
    
    def __init__(self, db_path: str = "scholar_results.db"):
        """
        Initialize unified database
        
        Args:
            db_path: Database file path
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Create scholar_results table (verification result cache)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scholar_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Basic information
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    venue TEXT,
                    url TEXT,
                    
                    -- Timestamp
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create search_results table (search result storage)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Basic information
                    title TEXT,
                    authors TEXT,
                    venue TEXT,
                    year INTEGER,
                    url TEXT,
                    
                    -- Search-related information
                    source TEXT NOT NULL,
                    search_query TEXT,
                    search_engine TEXT,
                    
                    -- Metadata
                    metadata TEXT,
                    result_position INTEGER,
                    
                    -- Timestamp
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for scholar_results table
            conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON scholar_results(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON scholar_results(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON scholar_results(created_at)")
            
            # Create unique constraint for scholar_results table
            try:
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_reference 
                    ON scholar_results(title, authors, year)
                """)
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    logger.warning("Duplicate records detected in scholar_results table, cleaning up...")
                    self._cleanup_scholar_duplicates(conn)
                    conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_reference 
                        ON scholar_results(title, authors, year)
                    """)
                    logger.info("Duplicate records in scholar_results cleaned up, unique index created successfully")
                else:
                    raise
            
            # Create index for search_results table
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_title ON search_results(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_authors ON search_results(authors)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_venue ON search_results(venue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_year ON search_results(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_source ON search_results(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON search_results(search_query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_engine ON search_results(search_engine)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_created_at ON search_results(created_at)")
            
            # Create unique constraint for search_results table
            # Main constraint: title + source must be unique (for the same document from the same source)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_search_result_basic 
                ON search_results(title, source)
                WHERE title IS NOT NULL AND source IS NOT NULL
            """)
            
            # Supplementary constraint: when complete information is available, title + authors + venue + source must be unique
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_search_result_full 
                ON search_results(title, authors, venue, source)
                WHERE title IS NOT NULL AND authors IS NOT NULL AND venue IS NOT NULL AND source IS NOT NULL
            """)
            
            conn.commit()
            logger.debug("Unified database tables created/verified")
    
    def _cleanup_scholar_duplicates(self, conn: sqlite3.Connection) -> int:
        """Clean up duplicate records in scholar_results table"""
        cursor = conn.execute("""
            DELETE FROM scholar_results 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM scholar_results 
                GROUP BY title, authors, year
            )
        """)
        deleted_count = cursor.rowcount
        logger.info(f"Cleaned up {deleted_count} duplicate records")
        return deleted_count
    
    # ========== Scholar Results related methods ==========
    
    def search_scholar_by_title(self, title: str) -> Optional[ScholarRecord]:
        """Search for scholar records by title"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM scholar_results 
                WHERE title LIKE ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (f"%{title}%",))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_scholar_record(row)
            return None
    
    def _row_to_scholar_record(self, row) -> ScholarRecord:
        """Convert database row to ScholarRecord object"""
        if not row:
            return None
        
        return ScholarRecord(
            id=row[0],
            title=row[1],
            authors=row[2],
            year=row[3],
            venue=row[4],
            url=row[5],
            created_at=datetime.fromisoformat(row[6]) if row[6] else None
        )
    
    def check_scholar_exists(self, title: str, authors: str, year: int) -> Optional[ScholarRecord]:
        """Check if scholar record exists"""
        if not title or not authors or not year:
            return None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM scholar_results 
                WHERE title = ? AND authors = ? AND year = ?
                LIMIT 1
            """, (title, authors, year))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_scholar_record(row)
            return None
    
    def insert_scholar_result(self, record: ScholarRecord, ignore_duplicates: bool = True) -> Optional[int]:
        """Insert scholar record"""
        if not ignore_duplicates:
            existing = self.check_scholar_exists(record.title, record.authors, record.year)
            if existing:
                raise ValueError(f"Duplicate scholar record: title='{record.title}', authors='{record.authors}', year={record.year}")
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO scholar_results (
                        title, authors, year, venue, url, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record.title, record.authors, record.year, record.venue, record.url, 
                    record.created_at or datetime.now()
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                logger.debug(f"Inserted scholar result with ID {record_id}: {record.title[:50] if record.title else 'No title'}...")
                return record_id
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    if ignore_duplicates:
                        logger.debug(f"Duplicate scholar record detected during insert, skipping: {record.title[:50] if record.title else 'No title'}...")
                        return None
                    else:
                        raise ValueError(f"Duplicate scholar record: title='{record.title}', authors='{record.authors}', year={record.year}") from e
                else:
                    raise
    
    def get_scholar_statistics(self) -> Dict[str, Any]:
        """Get scholar_results table statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total number of records
            cursor = conn.execute("SELECT COUNT(*) FROM scholar_results")
            total_records = cursor.fetchone()[0]
            
            return {
                'total_records': total_records
            }
    
    def cleanup_scholar_duplicates(self) -> int:
        """Clean up duplicate data in scholar_results table"""
        with sqlite3.connect(self.db_path) as conn:
            return self._cleanup_scholar_duplicates(conn)
    
    def clear_scholar_data(self):
        """Clear all data from scholar_results table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM scholar_results")
            conn.commit()
            logger.info("All scholar results data cleared")
    
    # ========== Search Results Related Methods ==========
    
    def check_search_duplicate(self, title: str, authors: str, venue: str, source: str) -> Optional[SearchResultRecord]:
        """Check if search result is duplicate"""
        # Can perform duplicate check as long as title and source are present
        if not title or not source:
            return None
        
        with sqlite3.connect(self.db_path) as conn:
            # Use basic title + source matching for duplicate check
            # This aligns with our primary uniqueness constraint
            cursor = conn.execute("""
                SELECT * FROM search_results 
                WHERE title = ? AND source = ?
                LIMIT 1
            """, (title, source))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_search_record(row)
            return None
    
    def _row_to_search_record(self, row) -> SearchResultRecord:
        """Convert database row to SearchResultRecord object"""
        if not row:
            return None
        
        return SearchResultRecord(
            id=row[0],
            title=row[1],
            authors=row[2],
            venue=row[3],
            year=row[4],
            url=row[5],
            source=row[6],
            search_query=row[7],
            search_engine=row[8],
            metadata=row[9],
            result_position=row[10],
            created_at=datetime.fromisoformat(row[11]) if row[11] else None
        )
    
    def insert_search_result(self, record: SearchResultRecord, ignore_duplicates: bool = True) -> Optional[int]:
        """Insert search result record"""
        if not ignore_duplicates:
            existing = self.check_search_duplicate(record.title, record.authors, record.venue, record.source)
            if existing:
                raise ValueError(f"Duplicate search result: title='{record.title}', source='{record.source}'")
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                now = datetime.now()
                cursor = conn.execute("""
                    INSERT INTO search_results (
                        title, authors, venue, year, url,
                        source, search_query, search_engine,
                        metadata, result_position, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.title, record.authors, record.venue, record.year, record.url,
                    record.source, record.search_query, record.search_engine,
                    record.metadata, record.result_position, record.created_at or now
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                logger.debug(f"Inserted search result with ID {record_id}: {record.title[:50] if record.title else 'No title'}...")
                return record_id
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    if ignore_duplicates:
                        logger.debug(f"Duplicate search result detected during insert, skipping: {record.title[:50] if record.title else 'No title'}...")
                        return None
                    else:
                        raise ValueError(f"Duplicate search result: title='{record.title}', source='{record.source}'") from e
                else:
                    raise
    
    def insert_search_results_batch(self, records: List[SearchResultRecord], ignore_duplicates: bool = True) -> Dict[str, int]:
        """Batch insert search result records"""
        stats = {
            'inserted': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        for record in records:
            try:
                record_id = self.insert_search_result(record, ignore_duplicates)
                if record_id:
                    stats['inserted'] += 1
                else:
                    stats['duplicates'] += 1
            except Exception as e:
                logger.error(f"Error inserting search result {record.title}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Batch insert completed: {stats}")
        return stats
    
    def search_search_results_by_title(self, title: str) -> List[SearchResultRecord]:
        """Search search result records by title"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM search_results 
                WHERE title LIKE ?
                ORDER BY created_at DESC
            """, (f"%{title}%",))
            
            return [self._row_to_search_record(row) for row in cursor.fetchall()]
    
    def search_search_results_by_source(self, source: str) -> List[SearchResultRecord]:
        """Search records by search source"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM search_results 
                WHERE source = ?
                ORDER BY created_at DESC
            """, (source,))
            
            return [self._row_to_search_record(row) for row in cursor.fetchall()]
    
    def get_search_results_statistics(self) -> Dict[str, Any]:
        """Get search results statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total number of records
            cursor = conn.execute("SELECT COUNT(*) FROM search_results")
            total_records = cursor.fetchone()[0]
            
            # Distribution by source
            cursor = conn.execute("""
                SELECT source, COUNT(*) 
                FROM search_results 
                GROUP BY source 
                ORDER BY COUNT(*) DESC
            """)
            source_distribution = dict(cursor.fetchall())
            
            # Distribution by search engine
            cursor = conn.execute("""
                SELECT search_engine, COUNT(*) 
                FROM search_results 
                WHERE search_engine IS NOT NULL 
                GROUP BY search_engine 
                ORDER BY COUNT(*) DESC
            """)
            search_engine_distribution = dict(cursor.fetchall())
            
            # Distribution by year
            cursor = conn.execute("""
                SELECT year, COUNT(*) 
                FROM search_results 
                WHERE year IS NOT NULL 
                GROUP BY year 
                ORDER BY year DESC
                LIMIT 10
            """)
            year_distribution = dict(cursor.fetchall())
            
            # Number of records with URL
            cursor = conn.execute("SELECT COUNT(*) FROM search_results WHERE url IS NOT NULL AND url != ''")
            records_with_url = cursor.fetchone()[0]
            
            # Most recent search record
            cursor = conn.execute("""
                SELECT created_at 
                FROM search_results 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            latest_result = cursor.fetchone()
            latest_search_time = latest_result[0] if latest_result else None
            
            return {
                'total_records': total_records,
                'source_distribution': source_distribution,
                'search_engine_distribution': search_engine_distribution,
                'year_distribution': year_distribution,
                'records_with_url': records_with_url,
                'latest_search_time': latest_search_time
            }
    
    def export_search_results_to_csv(self, output_path: str, source_filter: str = None) -> int:
        """Export search results to CSV file"""
        import pandas as pd
        
        where_clause = f"WHERE source = '{source_filter}'" if source_filter else ""
        
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT 
                    id, title, authors, venue, year, url,
                    source, search_query, search_engine,
                    metadata, result_position, created_at
                FROM search_results 
                {where_clause}
                ORDER BY created_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Exported {len(df)} search result records to {output_path}")
            return len(df)
    
    def clear_search_results_data(self):
        """Clear all search results data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM search_results")
            conn.commit()
            logger.info("All search results data cleared")
    
    # ========== General Methods ==========
    
    def clear_all_data(self):
        """Clear all data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM scholar_results")
            conn.execute("DELETE FROM search_results")
            conn.commit()
            logger.info("All data cleared from unified database")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None


def create_scholar_record_from_external_reference(external_ref: ExternalReference) -> ScholarRecord:
    """Create ScholarRecord from ExternalReference"""
    return ScholarRecord(
        title=external_ref.title,
        authors=', '.join(external_ref.authors) if external_ref.authors else None,
        year=external_ref.year,
        venue=external_ref.venue,
        url=external_ref.url,
        created_at=datetime.now()
    )


def create_search_result_record_from_external_reference(
    external_ref: ExternalReference, 
    search_query: str = None,
    result_position: int = None
) -> SearchResultRecord:
    """Create SearchResultRecord from ExternalReference"""
    authors_str = ', '.join(external_ref.authors) if external_ref.authors else None
    
    # Extract search engine information from metadata
    search_engine = None
    if external_ref.metadata:
        search_engine = external_ref.metadata.get('search_engine')
    
    # Serialize metadata
    metadata_json = None
    if external_ref.metadata:
        try:
            metadata_json = json.dumps(external_ref.metadata, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to serialize metadata: {e}")
    
    return SearchResultRecord(
        title=external_ref.title,
        authors=authors_str,
        venue=external_ref.venue,
        year=external_ref.year,
        url=external_ref.url,
        source=external_ref.source,
        search_query=search_query,
        search_engine=search_engine,
        metadata=metadata_json,
        result_position=result_position,
        created_at=datetime.now()
    )
