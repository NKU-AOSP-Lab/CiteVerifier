"""
Store parsed reference data
Support storage of original parsing results and LLM reparse results
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

from checker.models import Reference

logger = logging.getLogger(__name__)


@dataclass
class ParsedReferenceRecord:
    """Parsed reference record data structure"""
    id: Optional[int] = None
    
    # Basic information
    title: str = None
    authors: str = None  # Author list, separated by commas
    venue: str = None
    year: Optional[int] = None
    reference_type: str = None
    
    # Original data
    raw_text: str = None  # Original reference text
    
    # LLM reparse related
    is_llm_reparsed: bool = False  # Whether it has been LLM reparsed
    original_title: str = None  # Original title before LLM reparse
    original_authors: str = None  # Original authors before LLM reparse
    original_venue: str = None  # Original venue before LLM reparse
    original_year: Optional[int] = None  # Original year before LLM reparse
    
    # Metadata
    source_file: str = None  # Source file
    parser_version: str = "grobid"  # Parser version
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'venue': self.venue,
            'year': self.year,
            'reference_type': self.reference_type,
            'raw_text': self.raw_text,
            'is_llm_reparsed': self.is_llm_reparsed,
            'original_title': self.original_title,
            'original_authors': self.original_authors,
            'original_venue': self.original_venue,
            'original_year': self.original_year,
            'source_file': self.source_file,
            'parser_version': self.parser_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ParsedReferencesDatabase:
    """Parsed references database"""
    
    def __init__(self, db_path: str = "parsed_references.db"):
        """
        Initialize database
        
        Args:
            db_path: Database file path
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parsed_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Basic information
                    title TEXT,
                    authors TEXT,
                    venue TEXT,
                    year INTEGER,
                    reference_type TEXT,
                    
                    -- Original data
                    raw_text TEXT,
                    
                    -- LLM reparse related
                    is_llm_reparsed BOOLEAN DEFAULT FALSE,
                    original_title TEXT,
                    original_authors TEXT,
                    original_venue TEXT,
                    original_year INTEGER,
                    
                    -- Metadata
                    source_file TEXT,
                    parser_version TEXT DEFAULT 'grobid',
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # -- Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON parsed_references(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_authors ON parsed_references(authors)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_venue ON parsed_references(venue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON parsed_references(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON parsed_references(source_file)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_is_llm_reparsed ON parsed_references(is_llm_reparsed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON parsed_references(created_at)")
            
            # -- Create uniqueness constraint: title, authors, venue combination must be unique
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_reference 
                ON parsed_references(title, authors, venue)
                WHERE title IS NOT NULL AND authors IS NOT NULL AND venue IS NOT NULL
            """)
            
            conn.commit()
            logger.debug("Parsed references database tables created/verified")
    
    def check_duplicate(self, title: str, authors: str, venue: str) -> Optional[ParsedReferenceRecord]:
        """
        Check if duplicate record exists
        
        Args:
            title: Title
            authors: Authors (comma-separated)
            venue: Publication platform
            
        Returns:
            Return the record if it exists; otherwise return None
        """
        if not title or not authors or not venue:
            return None
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM parsed_references 
                WHERE title = ? AND authors = ? AND venue = ?
                LIMIT 1
            """, (title, authors, venue))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
            
            return None
    
    def _row_to_record(self, row) -> ParsedReferenceRecord:
        """Convert database row to record object"""
        if not row:
            return None
        
        return ParsedReferenceRecord(
            id=row[0],
            title=row[1],
            authors=row[2],
            venue=row[3],
            year=row[4],
            reference_type=row[5],
            raw_text=row[6],
            is_llm_reparsed=bool(row[7]),
            original_title=row[8],
            original_authors=row[9],
            original_venue=row[10],
            original_year=row[11],
            source_file=row[12],
            parser_version=row[13],
            created_at=datetime.fromisoformat(row[14]) if row[14] else None,
            updated_at=datetime.fromisoformat(row[15]) if row[15] else None
        )
    
    def insert_parsed_reference(self, record: ParsedReferenceRecord, ignore_duplicates: bool = True) -> Optional[int]:
        """
        Insert parsed reference record
        
        Args:
            record: Reference record
            ignore_duplicates: Whether to ignore duplicate records
            
        Returns:
            ID of inserted record, or None if duplicate and ignored
        """
        # Check for duplicates
        if not ignore_duplicates:
            existing = self.check_duplicate(record.title, record.authors, record.venue)
            if existing:
                raise ValueError(f"Duplicate reference: title='{record.title}', authors='{record.authors}', venue='{record.venue}'")
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                now = datetime.now()
                cursor = conn.execute("""
                    INSERT INTO parsed_references (
                        title, authors, venue, year, reference_type, raw_text,
                        is_llm_reparsed, original_title, original_authors, original_venue, original_year,
                        source_file, parser_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.title, record.authors, record.venue, record.year, record.reference_type, record.raw_text,
                    record.is_llm_reparsed, record.original_title, record.original_authors, 
                    record.original_venue, record.original_year,
                    record.source_file, record.parser_version, 
                    record.created_at or now, record.updated_at or now
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                logger.debug(f"Inserted parsed reference with ID {record_id}: {record.title[:50] if record.title else 'No title'}...")
                return record_id
            except sqlite3.IntegrityError as e:
                # Handle uniqueness constraint violation
                if "UNIQUE constraint failed" in str(e):
                    if ignore_duplicates:
                        logger.debug(f"Duplicate reference detected during insert, skipping: {record.title[:50] if record.title else 'No title'}...")
                        return None
                    else:
                        raise ValueError(f"Duplicate reference: title='{record.title}', authors='{record.authors}', venue='{record.venue}'") from e
                else:
                    raise
    
    def update_with_llm_reparse(self, record_id: int, llm_title: str, llm_authors: str, 
                               llm_venue: str, llm_year: Optional[int]) -> bool:
        """
        Update record with LLM reparse results
        
        Args:
            record_id: Record ID
            llm_title: LLM parsed title
            llm_authors: LLM parsed authors
            llm_venue: LLM parsed venue
            llm_year: LLM parsed year
            
        Returns:
            Whether update was successful
        """
        with sqlite3.connect(self.db_path) as conn:
            # First get original data
            cursor = conn.execute("""
                SELECT title, authors, venue, year FROM parsed_references WHERE id = ?
            """, (record_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Record with ID {record_id} not found")
                return False
            
            original_title, original_authors, original_venue, original_year = row
            
            # Update with LLM parse results
            cursor = conn.execute("""
                UPDATE parsed_references SET
                    title = ?,
                    authors = ?,
                    venue = ?,
                    year = ?,
                    is_llm_reparsed = TRUE,
                    original_title = ?,
                    original_authors = ?,
                    original_venue = ?,
                    original_year = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                llm_title, llm_authors, llm_venue, llm_year,
                original_title, original_authors, original_venue, original_year,
                datetime.now(), record_id
            ))
            
            conn.commit()
            logger.debug(f"Updated record {record_id} with LLM reparse results")
            return cursor.rowcount > 0
    
    def insert_batch(self, records: List[ParsedReferenceRecord], ignore_duplicates: bool = True) -> Dict[str, int]:
        """
        Batch insert parsed reference records
        
        Args:
            records: List of reference records
            ignore_duplicates: Whether to ignore duplicate records
            
        Returns:
            Insertion statistics
        """
        stats = {
            'inserted': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        for record in records:
            try:
                record_id = self.insert_parsed_reference(record, ignore_duplicates)
                if record_id:
                    stats['inserted'] += 1
                else:
                    stats['duplicates'] += 1
            except Exception as e:
                logger.error(f"Error inserting record {record.title}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Batch insert completed: {stats}")
        return stats
    
    def search_by_title(self, title: str) -> List[ParsedReferenceRecord]:
        """Search records by title"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM parsed_references 
                WHERE title LIKE ?
                ORDER BY created_at DESC
            """, (f"%{title}%",))
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def search_by_source_file(self, source_file: str) -> List[ParsedReferenceRecord]:
        """Search records by source file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM parsed_references 
                WHERE source_file = ?
                ORDER BY created_at DESC
            """, (source_file,))
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_llm_reparsed_records(self) -> List[ParsedReferenceRecord]:
        """Get all LLM reparsed records"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM parsed_references 
                WHERE is_llm_reparsed = TRUE
                ORDER BY created_at DESC
            """)
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total record count
            cursor = conn.execute("SELECT COUNT(*) FROM parsed_references")
            total_records = cursor.fetchone()[0]
            
            # LLM reparsed record count
            cursor = conn.execute("SELECT COUNT(*) FROM parsed_references WHERE is_llm_reparsed = TRUE")
            llm_reparsed_count = cursor.fetchone()[0]
            
            # Distribution by year
            cursor = conn.execute("""
                SELECT year, COUNT(*) 
                FROM parsed_references 
                WHERE year IS NOT NULL 
                GROUP BY year 
                ORDER BY year DESC
                LIMIT 10
            """)
            year_distribution = dict(cursor.fetchall())
            
            # Distribution by reference type
            cursor = conn.execute("""
                SELECT reference_type, COUNT(*) 
                FROM parsed_references 
                WHERE reference_type IS NOT NULL 
                GROUP BY reference_type 
                ORDER BY COUNT(*) DESC
            """)
            type_distribution = dict(cursor.fetchall())
            
            # Distribution by source file
            cursor = conn.execute("""
                SELECT source_file, COUNT(*) 
                FROM parsed_references 
                WHERE source_file IS NOT NULL 
                GROUP BY source_file 
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            source_file_distribution = dict(cursor.fetchall())
            
            return {
                'total_records': total_records,
                'llm_reparsed_count': llm_reparsed_count,
                'llm_reparsed_percentage': (llm_reparsed_count / total_records * 100) if total_records > 0 else 0,
                'year_distribution': year_distribution,
                'type_distribution': type_distribution,
                'source_file_distribution': source_file_distribution
            }
    
    def export_to_csv(self, output_path: str, include_llm_only: bool = False) -> int:
        """
        Export to CSV file
        
        Args:
            output_path: Output file path
            include_llm_only: Whether to include only LLM reparsed records
            
        Returns:
            Number of exported records
        """
        import pandas as pd
        
        where_clause = "WHERE is_llm_reparsed = TRUE" if include_llm_only else ""
        
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT 
                    id, title, authors, venue, year, reference_type,
                    raw_text, is_llm_reparsed,
                    original_title, original_authors, original_venue, original_year,
                    source_file, parser_version, created_at, updated_at
                FROM parsed_references 
                {where_clause}
                ORDER BY created_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Exported {len(df)} records to {output_path}")
            return len(df)
    
    def clear_all_data(self):
        """Clear all data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM parsed_references")
            conn.commit()
            logger.info("All parsed references data cleared")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None


def create_parsed_reference_record_from_dict(ref_dict: dict, source_file: str = None) -> ParsedReferenceRecord:
    """
    Create record from parse result dictionary
    
    Args:
        ref_dict: grobid parse result dictionary
        source_file: Source file name
        
    Returns:
        Parsed reference record
    """
    authors_str = ', '.join(ref_dict.get('authors', [])) if ref_dict.get('authors') else None
    
    return ParsedReferenceRecord(
        title=ref_dict.get('title'),
        authors=authors_str,
        venue=ref_dict.get('venue'),
        year=int(ref_dict.get('year')) if ref_dict.get('year') and str(ref_dict.get('year')).isdigit() else None,
        reference_type=ref_dict.get('reference_type'),
        raw_text=ref_dict.get('raw'),
        source_file=source_file,
        created_at=datetime.now()
    )


def create_parsed_reference_record_from_reference(ref: Reference, source_file: str = None) -> ParsedReferenceRecord:
    """
    Create record from Reference object
    
    Args:
        ref: Reference object
        source_file: Source file name
        
    Returns:
        Parsed reference record
    """
    authors_str = ', '.join(ref.authors) if ref.authors else None
    
    return ParsedReferenceRecord(
        title=ref.title,
        authors=authors_str,
        venue=ref.venue,
        year=ref.year,
        reference_type=ref.reference_type if hasattr(ref, 'reference_type') else None,
        raw_text=ref.raw if hasattr(ref, 'raw') else None,
        source_file=source_file,
        created_at=datetime.now()
    )
