"""
Reference storage service
Manages storage of parsed references and updating LLM reparse results
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from parsed_references_database import (
    ParsedReferencesDatabase, 
    ParsedReferenceRecord,
    create_parsed_reference_record_from_dict,
    create_parsed_reference_record_from_reference
)
from unified_database import (
    UnifiedDatabase,
    SearchResultRecord,
    create_search_result_record_from_external_reference
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checker.models import Reference, ExternalReference

logger = logging.getLogger(__name__)


class ReferenceStorageService:
    """Reference storage service (including parse results and search results)"""
    
    def __init__(self, db_path: str = "parsed_references.db", unified_db_path: str = "scholar_results.db"):
        """
        Initialize storage service
        
        Args:
            db_path: Parsed references database file path
            unified_db_path: Unified database file path (contains scholar_results and search_results)
        """
        self.parsed_refs_db = ParsedReferencesDatabase(db_path)
        self.unified_db = UnifiedDatabase(unified_db_path)
        
        self.stats = {
            # Parse result statistics
            'total_processed': 0,
            'inserted': 0,
            'duplicates': 0,
            'llm_reparsed': 0,
            'errors': 0,
            
            # Search result statistics
            'search_results_stored': 0,
            'search_results_duplicates': 0,
            'search_results_errors': 0
        }
    
    # Maintain backward compatibility property
    @property
    def db(self):
        """Backward compatibility: return parsed references database"""
        return self.parsed_refs_db
    
    def store_parsed_references_from_dict(self, references: List[dict], source_file: str = None) -> Dict[str, int]:
        """
        Store parsed results from dictionary format
        
        Args:
            references: grobid parse result dictionary list
            source_file: Source filename
            
        Returns:
            Storage statistics
        """
        logger.info(f"Starting to store {len(references)} parsed references, source file: {source_file}")
        
        records = []
        for ref_dict in references:
            try:
                record = create_parsed_reference_record_from_dict(ref_dict, source_file)
                # Only store records with titles
                if record.title and record.title.strip():
                    records.append(record)
                else:
                    logger.debug(f"Skipping reference without title: {ref_dict.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to create record: {ref_dict.get('id', 'unknown')}, error: {e}")
                self.stats['errors'] += 1
        
        if not records:
            logger.warning("No valid records to store")
            return {'inserted': 0, 'duplicates': 0, 'errors': self.stats['errors']}
        
        # Batch insert
        batch_stats = self.db.insert_batch(records, ignore_duplicates=True)
        
        # Update statistics
        self.stats['total_processed'] += len(references)
        self.stats['inserted'] += batch_stats['inserted']
        self.stats['duplicates'] += batch_stats['duplicates']
        self.stats['errors'] += batch_stats['errors']
        
        logger.info(f"Storage completed: {batch_stats}")
        return batch_stats
    
    def store_parsed_references_from_references(self, references: List[Reference], source_file: str = None) -> Dict[str, int]:
        """
        Store from Reference object list
        
        Args:
            references: Reference object list
            source_file: source filename
            
        Returns:
            store statistics
        """
        logger.info(f"Starting to store {len(references)} Reference object，source file: {source_file}")
        
        records = []
        for ref in references:
            try:
                record = create_parsed_reference_record_from_reference(ref, source_file)
                # Only store records with titles
                if record.title and record.title.strip():
                    records.append(record)
                else:
                    logger.debug(f"Skipping reference without title: {ref.id if hasattr(ref, 'id') else 'unknown'}")
            except Exception as e:
                logger.error(f"Failed to create record: {ref.id if hasattr(ref, 'id') else 'unknown'}, error: {e}")
                self.stats['errors'] += 1
        
        if not records:
            logger.warning("No valid records to store")
            return {'inserted': 0, 'duplicates': 0, 'errors': self.stats['errors']}
        
        # Batch insert
        batch_stats = self.db.insert_batch(records, ignore_duplicates=True)
        
        # update statistics
        self.stats['total_processed'] += len(references)
        self.stats['inserted'] += batch_stats['inserted']
        self.stats['duplicates'] += batch_stats['duplicates']
        self.stats['errors'] += batch_stats['errors']
        
        logger.info(f"Storage completed: {batch_stats}")
        return batch_stats
    
    def update_with_llm_reparse(self, original_ref: Reference, llm_reparsed_dict: dict) -> bool:
        """
        Update record with LLM reparse results
        
        Args:
            original_ref: Original Reference object
            llm_reparsed_dict: LLM reparse result dictionary
            
        Returns:
            Whether update successful
        """
        if not original_ref.title or not llm_reparsed_dict.get('title'):
            logger.warning("Original or LLM parse result missing title, cannot update")
            return False
        
        # Find original record
        original_authors = ', '.join(original_ref.authors) if original_ref.authors else None
        existing_record = self.db.check_duplicate(original_ref.title, original_authors, original_ref.venue)
        
        if not existing_record:
            logger.warning(f"Original record not found: {original_ref.title[:50]}...")
            return False
        
        # Prepare LLM parse results
        llm_authors = ', '.join(llm_reparsed_dict.get('authors', [])) if llm_reparsed_dict.get('authors') else None
        llm_year = int(llm_reparsed_dict.get('year')) if llm_reparsed_dict.get('year') and str(llm_reparsed_dict.get('year')).isdigit() else None
        
        # Update record
        success = self.db.update_with_llm_reparse(
            existing_record.id,
            llm_reparsed_dict.get('title'),
            llm_authors,
            llm_reparsed_dict.get('venue'),
            llm_year
        )
        
        if success:
            self.stats['llm_reparsed'] += 1
            logger.info(f"LLM reparse update successful: {existing_record.id} -> {llm_reparsed_dict.get('title')[:50]}...")
        
        return success
    
    def find_and_update_with_llm_reparse(self, llm_reparsed_dict: dict, source_file: str = None) -> bool:
        """
        Find and update record based on LLM reparse result
        
        Args:
            llm_reparsed_dict: LLM reparse result dictionary (Contains id field for matching)
            source_file: Source filename
            
        Returns:
            Whether update successful
        """
        ref_id = llm_reparsed_dict.get('id')
        if not ref_id:
            logger.warning("LLM reparse result missing ID, cannot match original record")
            return False
        
        # Find record by source file and ID
        if source_file:
            records = self.db.search_by_source_file(source_file)
            # Need to find based on actual ID matching logic here
            # Since database does not store original reference ID, we need other ways to match
            logger.warning("Currently cannot match records by ID, need to improve matching logic")
            return False
        
        return False
    
    def store_search_result(self, external_ref: ExternalReference, search_query: str = None, 
                           result_position: int = None) -> bool:
        """
        Store single search result
        
        Args:
            external_ref: External reference object
            search_query: search query
            result_position: Search result position
            
        Returns:
            Whether successfully stored
        """
        try:
            record = create_search_result_record_from_external_reference(
                external_ref, search_query, result_position
            )
            
            record_id = self.unified_db.insert_search_result(record, ignore_duplicates=True)
            
            if record_id:
                self.stats['search_results_stored'] += 1
                logger.debug(f"Stored search result successfully: {external_ref.title[:50] if external_ref.title else 'No title'}...")
                return True
            else:
                self.stats['search_results_duplicates'] += 1
                logger.debug(f"Search result duplicate, skipping: {external_ref.title[:50] if external_ref.title else 'No title'}...")
                return False
                
        except Exception as e:
            self.stats['search_results_errors'] += 1
            logger.error(f"Failed to store search result: {external_ref.title if external_ref.title else 'No title'}, error: {e}")
            return False
    
    def store_search_results_batch(self, external_refs: List[ExternalReference], 
                                  search_query: str = None) -> Dict[str, int]:
        """
        Batch store search results
        
        Args:
            external_refs: External reference object list
            search_query: search query
            
        Returns:
            store statistics
        """
        logger.info(f"Starting batch storage {len(external_refs)} search results")
        
        records = []
        for i, external_ref in enumerate(external_refs):
            try:
                record = create_search_result_record_from_external_reference(
                    external_ref, search_query, i + 1
                )
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to create search result record: {external_ref.title if external_ref.title else 'No title'}, error: {e}")
                self.stats['search_results_errors'] += 1
        
        if not records:
            logger.warning("No valid search result records to store")
            return {'inserted': 0, 'duplicates': 0, 'errors': self.stats['search_results_errors']}
        
        # Batch insert
        batch_stats = self.unified_db.insert_search_results_batch(records, ignore_duplicates=True)
        
        # update statistics
        self.stats['search_results_stored'] += batch_stats['inserted']
        self.stats['search_results_duplicates'] += batch_stats['duplicates']
        self.stats['search_results_errors'] += batch_stats['errors']
        
        logger.info(f"Search result storage completed: {batch_stats}")
        return batch_stats
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """getstore statistics"""
        parsed_refs_stats = self.parsed_refs_db.get_statistics()
        search_results_stats = self.unified_db.get_search_results_statistics()
        
        return {
            **self.stats,
            'parsed_refs': parsed_refs_stats,
            'search_results': search_results_stats
        }
    
    def print_storage_statistics(self):
        """Print storage statistics"""
        stats = self.get_storage_statistics()
        
        print("\n" + "="*60)
        print("Reference storage statistics")
        print("="*60)
        
        # Parse result statistics
        print("Parse result storage:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successfully inserted: {stats['inserted']}")
        print(f"  Duplicates skipped: {stats['duplicates']}")
        print(f"  LLM reparse: {stats['llm_reparsed']}")
        print(f"  Errors: {stats['errors']}")
        
        # search resultsstatistics
        print("\nSearch result storage:")
        print(f"  stored: {stats['search_results_stored']}")
        print(f"  duplicates skipped: {stats['search_results_duplicates']}")
        print(f"  errors: {stats['search_results_errors']}")
        
        # Parsed references database statistics
        parsed_refs_stats = stats.get('parsed_refs', {})
        print("\nParsed references database statistics:")
        print(f"  total records: {parsed_refs_stats.get('total_records', 0)}")
        print(f"  LLM reparsed record count: {parsed_refs_stats.get('llm_reparsed_count', 0)}")
        print(f"  LLM reparse percentage: {parsed_refs_stats.get('llm_reparsed_percentage', 0):.1f}%")
        
        # Search results database statistics
        search_results_stats = stats.get('search_results', {})
        print("\nSearch results database statistics:")
        print(f"  total records: {search_results_stats.get('total_records', 0)}")
        print(f"  Records with URL: {search_results_stats.get('records_with_url', 0)}")
        
        # Search source distribution
        if search_results_stats.get('source_distribution'):
            print("\nSearch source distribution:")
            for source, count in search_results_stats['source_distribution'].items():
                print(f"  {source}: {count}")
        
        # Search engine distribution
        if search_results_stats.get('search_engine_distribution'):
            print("\nSearch engine distribution:")
            for engine, count in search_results_stats['search_engine_distribution'].items():
                print(f"  {engine}: {count}")
        
        # Reference type distribution
        if parsed_refs_stats.get('type_distribution'):
            print("\nReference type distribution:")
            for ref_type, count in list(parsed_refs_stats['type_distribution'].items())[:5]:
                print(f"  {ref_type}: {count}")
        
        # Year distribution
        if parsed_refs_stats.get('year_distribution'):
            print("\nYear distribution (top 5 years):")
            for year, count in list(parsed_refs_stats['year_distribution'].items())[:5]:
                print(f"  {year}: {count}")
        
        print("="*60)
    
    def export_to_csv(self, output_path: str, include_llm_only: bool = False) -> int:
        """
        Export parsed results to CSV file
        
        Args:
            output_path: Output file path
            include_llm_only: Whether to include only LLM reparsed records
            
        Returns:
            Number of exported records
        """
        return self.parsed_refs_db.export_to_csv(output_path, include_llm_only)
    
    def export_search_results_to_csv(self, output_path: str, source_filter: str = None) -> int:
        """
        Export search results to CSV file
        
        Args:
            output_path: Output file path
            source_filter: Source filter (e.g., 'google_search' or 'scrapingdog')
            
        Returns:
            Number of exported records
        """
        return self.unified_db.export_search_results_to_csv(output_path, source_filter)
    
    def get_llm_reparsed_records(self) -> List[ParsedReferenceRecord]:
        """Get all LLM reparsed records"""
        return self.db.get_llm_reparsed_records()
    
    def search_by_title(self, title: str) -> List[ParsedReferenceRecord]:
        """Search records by title"""
        return self.db.search_by_title(title)
    
    def search_by_source_file(self, source_file: str) -> List[ParsedReferenceRecord]:
        """Search records by source file"""
        return self.db.search_by_source_file(source_file)
    
    def close(self):
        """Close database connection"""
        self.parsed_refs_db.close()
        self.unified_db.close()


class StorageIntegratedFileProcessor:
    """File processor with integrated storage functionality"""
    
    def __init__(self, verifier, storage_service: ReferenceStorageService):
        self.verifier = verifier
        self.storage_service = storage_service
    
    async def process_directory_with_storage(self, dir_path: str, args) -> Dict[str, List[Reference]]:
        """Process files in directory and store parse results"""
        import grobid_parser_to_xml
        from tqdm import tqdm
        from checker.models import convert_parsed_references
        
        references = {}
       
        output_dir = dir_path.strip('/').split('/')[-1] + '_ref_xml'
        print(f"File path {dir_path}, xml output path{output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        files = os.listdir(output_dir)
        exist_files = [f for f in files if f.endswith('.xml')]
        exist_files_str = ','.join(exist_files)

        files = os.listdir(dir_path)
        
        for file in tqdm(files):
            if file.endswith('.pdf') and file not in exist_files_str:
                print(f"Processing {file}")
                try:
                    # Parse PDF
                    parsed_refs = grobid_parser_to_xml.grobid_parse(os.path.join(dir_path, file), output_dir)
                    
                    # Store parse results
                    storage_stats = self.storage_service.store_parsed_references_from_dict(parsed_refs, file)
                    logger.info(f"file {file} Storage statistics: {storage_stats}")
                    
                    # Convert to Reference object for subsequent verification
                    references[file] = convert_parsed_references(parsed_refs)
                    
                except Exception as e:
                    print(f"Process {file} failed: {e}")
                    logger.error(f"Processing file {file} failed: {e}")
                    continue
        
        total = len(references)
        print(f"Loaded {total} reference files")
        references = self._exclude_reference_type(references)
        
        return references
    
    async def process_single_file_with_storage(self, file_path: str) -> List[Reference]:
        """Process single file and store parse results"""
        from parser.grobid_parser import parse_xml
        from checker.models import convert_parsed_references
        
        file_name = Path(file_path).name
        
        # parse file
        parsed_refs = await self._exclude_no_venue(parse_xml(file_path))
        
        # Store parse results
        storage_stats = self.storage_service.store_parsed_references_from_dict(parsed_refs, file_name)
        logger.info(f"file {file_name} Storage statistics: {storage_stats}")
        
        return convert_parsed_references(parsed_refs)
    
    async def _exclude_no_venue(self, references: List[dict]) -> List[dict]:
        """Filter out references that do not meet criteria, and perform LLM reparse on references with empty titles"""
        references_ = []
        for ref in references:
            if ref['venue'] != 'monograph' and ref['venue'] != 'unknown':
                references_.append(ref)
        
        # Perform LLM reparse on references with empty titles
        for i in range(len(references_)):
            if references_[i]['title'] is None or references_[i]['title'].strip() == '':
                logger.info(f"Found reference with empty title, attempting LLM reparse: {references_[i]['id']}")
                reparsed_dict = await self.verifier.llm_reparser.reparse_with_llm(references_[i])
                if reparsed_dict and reparsed_dict.get('title'):
                    # Update stored record
                    references_[i] = reparsed_dict
                    logger.info(f"LLM reparse successful: {reparsed_dict['title'][:50]}...")
                else:
                    logger.warning(f"LLM reparse failed: {references_[i]['id']}")
        
        return references_
    
    def _exclude_reference_type(self, references: Dict[str, List[Reference]]) -> Dict[str, List[Reference]]:
        """Filter out references that do not meet criteria"""
        references_ = {}
        for name, ref in references.items():
            refs = []
            for r in ref:
                if r.reference_type != 'unknown' and r.reference_type != 'monograph':
                    refs.append(r)
            references_[name] = refs
        return references_
