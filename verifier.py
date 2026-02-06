#!/usr/bin/env python3

import asyncio
import logging
from math import log
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.llm_parser import llm_str2ref
from checker.models import Reference, ExternalReference, convert_parsed_reference, convert_parsed_references, VerificationResult, VerificationStatus
from checker.clients.scrapingdog_client import ScrapingDogClient
from checker.clients.google_search_client import GoogleSearchClient
from checker.utils import StringUtils, AuthorUtils
from unified_database import (
    UnifiedDatabase, 
    ScholarRecord, 
    create_scholar_record_from_external_reference
)
from reference_storage_service import ReferenceStorageService, StorageIntegratedFileProcessor
import csv

from checker.logger_config import setup_logging

# Set up logging
log_filename = setup_logging(log_to_file=True, log_level="DEBUG")
if log_filename:
    print(f"Logs will be saved to: {log_filename}")

logger = logging.getLogger(__name__)


@dataclass
class VerificationStats:
    """Verification statistics"""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    total_results_found: int = 0
    verified_valid: int = 0
    verified_invalid: int = 0
    verified_suspicious: int = 0
    verified_unverified: int = 0
    verified_error: int = 0
    google_fallback_used: int = 0
    google_fallback_successful: int = 0
    llm_reparse_used: int = 0
    llm_reparse_successful: int = 0
    total_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def update_verification_status(self, status: VerificationStatus):
        """Update verification status statistics"""
        if status == VerificationStatus.VALID:
            self.verified_valid += 1
        elif status == VerificationStatus.INVALID:
            self.verified_invalid += 1
        elif status == VerificationStatus.SUSPICIOUS:
            self.verified_suspicious += 1
        elif status == VerificationStatus.UNVERIFIED:
            self.verified_unverified += 1
        elif status == VerificationStatus.ERROR:
            self.verified_error += 1

    def print_statistics(self):
        """Print statistics"""
        print("\n" + "="*60)
        print("Verification Statistics")
        print("="*60)
        print(f"Total verifications: {self.total_searches}")
        print(f"Successful searches: {self.successful_searches}")
        print(f"Failed searches: {self.failed_searches}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"API calls: {self.api_calls}")
        print(f"Total results found: {self.total_results_found}")
        print()
        print("Verification result distribution:")
        print(f"  Valid (VALID): {self.verified_valid}")
        print(f"  Invalid (INVALID): {self.verified_invalid}")
        print(f"  Suspicious (SUSPICIOUS): {self.verified_suspicious}")
        print(f"  Unverified (UNVERIFIED): {self.verified_unverified}")
        print(f"  Error (ERROR): {self.verified_error}")
        print()
        print("Google search fallback statistics:")
        print(f"  Used count: {self.google_fallback_used}")
        print(f"  Successful count: {self.google_fallback_successful}")
        print()
        print("LLM reparse statistics:")
        print(f"  Used count: {self.llm_reparse_used}")
        print(f"  Successful count: {self.llm_reparse_successful}")
        print(f"Total time: {self.total_time:.2f} seconds")
        
        if self.total_searches > 0:
            success_rate = (self.successful_searches / self.total_searches) * 100
            cache_rate = (self.cache_hits / self.total_searches) * 100
            valid_rate = (self.verified_valid / self.total_searches) * 100
            avg_time = self.total_time / self.total_searches
            print(f"Search success rate: {success_rate:.1f}%")
            print(f"Cache hit rate: {cache_rate:.1f}%")
            print(f"Verification valid rate: {valid_rate:.1f}%")
            if self.google_fallback_used > 0:
                google_success_rate = (self.google_fallback_successful / self.google_fallback_used) * 100
                print(f"Google fallback success rate: {google_success_rate:.1f}%")
            if self.llm_reparse_used > 0:
                llm_success_rate = (self.llm_reparse_successful / self.llm_reparse_used) * 100
                print(f"LLM reparse success rate: {llm_success_rate:.1f}%")
            print(f"Average time: {avg_time:.2f} seconds/item")
        
        print("="*60)


class VerificationStrategy:
    """Verification strategy base class"""
    
    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        """Execute verification"""
        raise NotImplementedError


class DblpVerificationStrategy(VerificationStrategy):
    """Local DBLP matching strategy (runs before online matching)."""

    def __init__(
        self,
        validator: 'ReferenceValidator',
        dblp_db_path: Optional[str],
        dblp_match_threshold: float = 0.9,
        max_candidates: int = 100000,
    ):
        self.validator = validator
        self.dblp_db_path = Path(dblp_db_path) if dblp_db_path else None
        self.dblp_match_threshold = dblp_match_threshold
        self.max_candidates = max_candidates
        self._use_index: Optional[bool] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._all_titles: Optional[List[Any]] = None

    def _ensure_ready(self) -> bool:
        if not self.dblp_db_path or not self.dblp_db_path.exists():
            return False

        if self._use_index is None:
            # Decide the fastest strategy once: indexed search or full DB load.
            from dblp_match import _db_has_word_index, load_all_titles_from_db

            conn = sqlite3.connect(str(self.dblp_db_path))
            self._use_index = _db_has_word_index(conn)
            conn.close()

            if not self._use_index:
                self._all_titles = load_all_titles_from_db(self.dblp_db_path)

        if self._use_index and self._conn is None:
            # Keep a read-only connection for low-memory indexed search.
            from dblp_match import _sqlite_readonly_fast

            self._conn = sqlite3.connect(str(self.dblp_db_path))
            _sqlite_readonly_fast(self._conn)

        return True

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        if not reference.title:
            return None
        if not self._ensure_ready():
            return None

        from dblp_match import search_dblp_by_index, search_dblp_brute_force

        if self._use_index:
            assert self._conn is not None
            # Indexed search: fast candidate selection + ratio matching.
            match = search_dblp_by_index(self._conn, reference.title, self.max_candidates)
        else:
            assert self._all_titles is not None
            # Brute force: ratio matching over all titles (memory-heavy fallback).
            match = search_dblp_brute_force(self._all_titles, reference.title)

        if not match:
            return None

        external_ref = ExternalReference(
            title=match.get("dblp_title"),
            authors=None,
            year=None,
            venue=None,
            url=None,
            source="dblp",
            metadata={
                "dblp_id": match.get("dblp_id"),
                "dblp_title_similarity": match.get("dblp_title_similarity"),
            },
        )

        result = self.validator.validate_external_reference(reference, external_ref, "DBLP match")
        if match.get("dblp_title_similarity", 0.0) < self.dblp_match_threshold:
            result.verification_notes.append(
                f"DBLP title similarity below threshold: {match.get('dblp_title_similarity', 0.0):.2f} < {self.dblp_match_threshold:.2f}"
            )
        return result


class CacheVerificationStrategy(VerificationStrategy):
    """Cache verification strategy"""
    
    def __init__(self, db: UnifiedDatabase, validator: 'ReferenceValidator'):
        self.db = db
        self.validator = validator
    
    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        """Cache verification"""
        cached_result = self.db.search_scholar_by_title(reference.title)
        if cached_result:
            logger.info(f"Reference {reference.id} found in cache, performing verification...")
            external_ref = self._create_external_ref_from_record(cached_result)
            return self.validator.validate_external_reference(reference, external_ref, "Cache verification")
        return None
    
    def _create_external_ref_from_record(self, record: ScholarRecord):
        """Create ExternalReference from database record"""
        from checker.models import ExternalReference
        
        return ExternalReference(
            title=record.title,
            authors=record.authors.split(', ') if record.authors else [],
            year=record.year,
            venue=record.venue,
            url=record.url,
            source="scrapingdog"
        )


class APIVerificationStrategy(VerificationStrategy):
    """API verification strategy"""
    
    def __init__(self, validator: 'ReferenceValidator', storage_service: 'ReferenceStorageService' = None):
        self.validator = validator
        self.storage_service = storage_service
    
    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify through API"""
        scrapingdog_client = clients.get('scrapingdog')
        if scrapingdog_client:
            try:
                external_ref = await scrapingdog_client.search_reference(reference)
                if external_ref:
                    # Store search results
                    if self.storage_service:
                        search_query = reference.title if reference.title else ""
                        self.storage_service.store_search_result(external_ref, search_query, 1)
                    
                    return self.validator.validate_external_reference(reference, external_ref, "ScrapingDog verification")
            except Exception as e:
                logger.warning(f"ScrapingDog searchfailed: {reference.id}, error: {e}")
        return None


class GoogleFallbackStrategy(VerificationStrategy):
    """Google search fallback strategy"""
    
    def __init__(self, validator: 'ReferenceValidator', storage_service: 'ReferenceStorageService' = None):
        self.validator = validator
        self.storage_service = storage_service
    
    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        """Google search fallback verification"""
        google_client = clients.get('google')
        if google_client:
            try:
                external_ref = await google_client.search_reference(reference)
                if external_ref:
                    # Store search results
                    if self.storage_service:
                        search_query = reference.title if reference.title else ""
                        self.storage_service.store_search_result(external_ref, search_query, 1)
                    
                    return self.validator.validate_external_reference(reference, external_ref, "Google search verification")
            except Exception as e:
                logger.warning(f"Google searchfailed: {reference.id}, error: {e}")
        return None


class LLMReparseStrategy(VerificationStrategy):
    """LLM reparse strategy"""
    
    def __init__(self, validator: 'ReferenceValidator', llm_reparser: 'LLMReparser', 
                 storage_service: 'ReferenceStorageService' = None):
        self.validator = validator
        self.llm_reparser = llm_reparser
        self.storage_service = storage_service
    
    async def verify(self, reference: Reference, clients: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify after LLM reparse"""
        reparsed_dict = await self.llm_reparser.reparse_with_llm(reference)
        if reparsed_dict:
            reparsed_reference = convert_parsed_reference(reparsed_dict, reparsed_dict['id'])
            if self.storage_service:
                self._store_llm_reparsed_result(reference, reparsed_dict)
            if reparsed_reference:
                # Try ScrapingDog first, then Google
                for client_name, method_suffix in [('scrapingdog', 'ScrapingDog'), ('google', 'Google search')]:
                    client = clients.get(client_name)
                    if client:
                        try:
                            external_ref = await client.search_reference(reparsed_reference)
                            if external_ref:
                                # Store search results
                                if self.storage_service:
                                    search_query = reparsed_reference.title if reparsed_reference.title else ""
                                    self.storage_service.store_search_result(external_ref, search_query, 1)
                                
                                result = self.validator.validate_external_reference(
                                    reparsed_reference, external_ref, f"{method_suffix}LLM reparse+verify"
                                )

                                if result.final_status == VerificationStatus.VALID:
                                    result.sources_checked = [client_name, "llm_reparse"]
                                    result.verification_notes.append("Verification successful after LLM reparse")
                                    
                        
                                    
                                    return result
                                else:
                                    result.sources_checked = [client_name, "llm_reparse"]
                                    result.verification_notes.append("Verification failed after LLM reparse")
                                    return result
                        except Exception as e:
                            logger.warning(f"{method_suffix}searchfailed: {reference.id}, error: {e}")
        return None
    
    def _store_llm_reparsed_result(self, original_reference: Reference, reparsed_dict: dict):
        """
        Store LLM reparse results
        
        Args:
            original_reference: Original reference
            reparsed_dict: Dictionary result after LLM reparse
        """
        try:
            from parsed_references_database import ParsedReferenceRecord
            
            # Create LLM reparsed record
            record = ParsedReferenceRecord(
                # Use reparsed information as primary information
                title=reparsed_dict.get('title'),
                authors=', '.join(reparsed_dict.get('authors', [])) if reparsed_dict.get('authors') else None,
                venue=reparsed_dict.get('venue'),
                year=reparsed_dict.get('year'),
                reference_type=reparsed_dict.get('reference_type', 'unknown'),
                raw_text=reparsed_dict.get('raw'),
                
                # LLM reparse flag
                is_llm_reparsed=True,
                
                # Save original parse information
                original_title=original_reference.title,
                original_authors=', '.join(original_reference.authors) if original_reference.authors else None,
                original_venue=original_reference.venue,
                original_year=original_reference.year,
                
                # Metadata
                source_file=f"llm_reparse_{original_reference.id}",
                parser_version="llm_reparse"
            )
            
            # Store record
            record_id = self.storage_service.parsed_refs_db.insert_parsed_reference(record, ignore_duplicates=True)
            
            if record_id:
                self.storage_service.stats['llm_reparsed'] += 1
                logger.info(f"Stored LLM reparse results: {original_reference.id} -> {reparsed_dict.get('title', '')[:50]}...")
            else:
                logger.debug(f"LLM reparse result duplicate, skipping storage: {original_reference.id}")
                
        except Exception as e:
            logger.error(f"Failed to store LLM reparse results: {original_reference.id}, error: {e}")
            self.storage_service.stats['errors'] += 1


class ReferenceValidator:
    """Reference validator"""
    
    def validate_external_reference(self, input_ref: Reference, external_ref, method: str = "APIsearch") -> VerificationResult:
        """Validate external reference information"""
        start_time = time.time()
        
        # 1. Title match check
        title_similarity = 0.0
        problematic_fields = []
        
        if input_ref.title and external_ref.title:
            # Use enhanced title similarity calculation
            title_similarity = StringUtils.enhanced_title_similarity(input_ref.title, external_ref.title)
            
            if title_similarity <= 0.9:  # Title similarity threshold
                return VerificationResult(
                    reference_id=input_ref.id,
                    final_status=VerificationStatus.INVALID,
                    diagnosis="Serious title error",
                    problematic_fields=["title"],
                    best_match=external_ref,
                    sources_checked=[external_ref.source] if external_ref.source else ["unknown"],
                    total_time=time.time() - start_time,
                    verification_notes=[f"Title similarity: {title_similarity:.2f}"],
                    recommendations=["Check if title is correct"]
                )
        
        # Comprehensive judgment (simplified version)
        diagnosis = "Verification passed"
        status = VerificationStatus.VALID
        
        return VerificationResult(
            reference_id=input_ref.id,
            final_status=status,
            diagnosis=diagnosis,
            problematic_fields=problematic_fields,
            best_match=external_ref,
            sources_checked=[external_ref.source] if external_ref.source else ["unknown"],
            total_time=time.time() - start_time,
            verification_notes=[f"Title similarity: {title_similarity:.2f}"],
            recommendations=self._generate_recommendations(status, problematic_fields)
        )
    
    def _generate_recommendations(self, status: VerificationStatus, problematic_fields: list) -> list:
        """Generate recommendations"""
        recommendations = []
        
        if status == VerificationStatus.VALID:
            recommendations.append("Reference verification passed")
        elif status == VerificationStatus.SUSPICIOUS:
            if "title" in problematic_fields:
                recommendations.append("Check if title is correct")
            if "authors" in problematic_fields:
                recommendations.append("Check author information")
            if "year" in problematic_fields:
                recommendations.append("Check publication year")
        elif status == VerificationStatus.INVALID:
            recommendations.append("Reference information error, needs correction")
        else:
            recommendations.append("Unable to verify reference")
        
        return recommendations


class LLMReparser:
    """LLM reparser"""
    
    async def reparse_with_llm(self, reference) -> Optional[dict]:
        """
        Use LLM to reparse original text of reference
        
        Args:
            reference: Original reference object, can be Reference type or dictionary type
            
        Returns:
            Reparsed reference dictionary, return None if parsing fails
        """
        # Handle different input types
        if isinstance(reference, dict):
            ref_id = reference.get('id')
            raw_text = reference.get('raw')
            ref_type = reference.get('reference_type', 'unknown')
        else:
            ref_id = reference.id
            raw_text = reference.raw
            ref_type = reference.reference_type if hasattr(reference, 'reference_type') else 'unknown'
        
        if not raw_text:
            logger.warning(f"Reference {ref_id} has no original text, cannot perform LLM reparse")
            return None
        
        try:
            logger.info(f"Reparse reference using LLM {ref_id}: {raw_text[:50]}...")
            
            # Use async semaphore to limit concurrency
            semaphore = asyncio.Semaphore(1)  # Limit LLM call concurrency
            
            # Call LLM parsing
            parsed_data = await llm_str2ref(raw_text, semaphore)
            
            if not parsed_data or not parsed_data.get('title'):
                logger.warning(f"LLM parse result empty or missing title: {ref_id}")
                return None
            
            # Preserve original information
            parsed_data['id'] = ref_id
            parsed_data['raw'] = raw_text
            parsed_data['reference_type'] = ref_type
            
            logger.info(f"LLM reparse successful: {ref_id} -> {parsed_data['title'][:50]}...")
            return parsed_data
            
        except Exception as e:
            logger.error(f"LLM reparse failed: {ref_id}, error: {e}")
            return None


class VerificationChain:
    """Verification chain - Execute different verification strategies in order"""
    
    def __init__(self, strategies: List[VerificationStrategy]):
        self.strategies = strategies
    
    async def execute(self, reference: Reference, clients: Dict[str, Any], stats: VerificationStats) -> VerificationResult:
        """Execute verification chain"""
        start_time = time.time()
        sources_attempted = []
        verification_notes = []
        last_external_ref = None  # Save last external_ref

        for strategy in self.strategies:
            try:
                result = await strategy.verify(reference, clients)
                if result:
                    # Save last external_ref (even if verification fails)
                    if result.best_match:
                        last_external_ref = result.best_match

                    if result.final_status == VerificationStatus.VALID:
                        # Update statistics
                        if isinstance(strategy, CacheVerificationStrategy):
                            stats.cache_hits += 1
                        elif isinstance(strategy, APIVerificationStrategy):
                            stats.api_calls += 1
                            stats.successful_searches += 1
                            stats.total_results_found += 1
                        elif isinstance(strategy, GoogleFallbackStrategy):
                            stats.google_fallback_used += 1
                            stats.google_fallback_successful += 1
                        elif isinstance(strategy, LLMReparseStrategy):
                            stats.llm_reparse_used += 1
                            stats.llm_reparse_successful += 1

                        stats.update_verification_status(result.final_status)
                        return result
                    else:
                        # Record attempted strategies
                        if isinstance(strategy, GoogleFallbackStrategy):
                            stats.google_fallback_used += 1
                            sources_attempted.append("google_search")
                        elif isinstance(strategy, LLMReparseStrategy):
                            stats.llm_reparse_used += 1
                            sources_attempted.append("llm_reparse")

                        verification_notes.extend(result.verification_notes or [])
            except Exception as e:
                logger.error(f"Verification strategy execution failed: {type(strategy).__name__}, error: {e}")
                verification_notes.append(f"{type(strategy).__name__}error: {str(e)}")

        # All strategies failed, return last external_ref instead of None
        stats.failed_searches += 1
        result = VerificationResult(
            reference_id=reference.id,
            final_status=VerificationStatus.INVALID,
            diagnosis="All verification methods failed to verify reference",
            problematic_fields=[],
            best_match=last_external_ref,  # Use last external_ref
            sources_checked=sources_attempted or ["scrapingdog"],
            total_time=time.time() - start_time,
            verification_notes=verification_notes,
            recommendations=["Check if reference information is correct, may need manual verification"]
        )
        stats.update_verification_status(result.final_status)
        return result


class ScrapingDogVerifier:
    """ScrapingDog batch verifier"""
    
    def __init__(
        self,
        db_path: str = "scholar_results.db",
        max_concurrent: int = 3,
        parsed_refs_db_path: str = "parsed_references.db",
        dblp_db_path: Optional[str] = "dblp_titles.db",
        dblp_match_threshold: float = 0.9,
        dblp_max_candidates: int = 100000,
        enable_dblp: bool = True,
    ):
        """
        Initialize verifier
        
        Args:
            db_path: Unified database file path (contains scholar_results and search_results tables)
            max_concurrent: Maximum concurrency
            parsed_refs_db_path: Parsed references database file path
        """
        self.db = UnifiedDatabase(db_path)
        self.max_concurrent = max_concurrent
        self.client: Optional[ScrapingDogClient] = None
        self.google_client: Optional[GoogleSearchClient] = None
        self.stats = VerificationStats()
        
        # Initialize components
        self.validator = ReferenceValidator()
        self.llm_reparser = LLMReparser()
        
        # Initialize storage service
        self.storage_service = ReferenceStorageService(parsed_refs_db_path, db_path)

        # DBLP matching configuration
        self.dblp_db_path = dblp_db_path
        self.dblp_match_threshold = dblp_match_threshold
        self.dblp_max_candidates = dblp_max_candidates
        self.enable_dblp = enable_dblp
        self.dblp_strategy: Optional[DblpVerificationStrategy] = None
        
        # Build verification strategy chain
        self._build_verification_chain()
    
    def _build_verification_chain(self):
        """Build verification strategy chain"""
        strategies: List[VerificationStrategy] = []

        # DBLP match first (local, ratio-based)
        if self.enable_dblp:
            self.dblp_strategy = DblpVerificationStrategy(
                self.validator,
                self.dblp_db_path,
                self.dblp_match_threshold,
                self.dblp_max_candidates,
            )
            strategies.append(self.dblp_strategy)

        strategies.extend([
            CacheVerificationStrategy(self.db, self.validator),
            APIVerificationStrategy(self.validator, self.storage_service),
            GoogleFallbackStrategy(self.validator, self.storage_service),
            LLMReparseStrategy(self.validator, self.llm_reparser, self.storage_service),
        ])

        self.verification_chain = VerificationChain(strategies)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = ScrapingDogClient()
        await self.client.initialize()
        
        self.google_client = GoogleSearchClient()
        await self.google_client.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()
        if self.google_client:
            await self.google_client.close()
        if self.storage_service:
            self.storage_service.close()
        if self.dblp_strategy:
            self.dblp_strategy.close()
    
    async def verify_single_reference(self, reference: Reference) -> VerificationResult:
        """
        Verify single reference
        
        Args:
            reference: Reference to be verified
            
        Returns:
            Verification result
        """
        logger.info(f"Verifying reference {reference.id}: {reference.title[:50]}...")
        
        clients = {
            'scrapingdog': self.client,
            'google': self.google_client
        }
        
        result = await self.verification_chain.execute(reference, clients, self.stats)
        
        # If verification passes, store to database cache
        if result.final_status == VerificationStatus.VALID and result.best_match:
            record = create_scholar_record_from_external_reference(result.best_match)
            self.db.insert_scholar_result(record)
        
        return result
    
    async def verify_batch(self, references: List[Reference]) -> List[VerificationResult]:
        """
        Batch verify references
        
        Args:
            references: List of references to be verified
            
        Returns:
            List of verification results
        """
        self.stats.start_time = time.time()
        self.stats.total_searches = len(references)
        
        logger.info(f"Starting batch verification {len(references)} references, maximum concurrency: {self.max_concurrent}")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def verify_with_semaphore(ref: Reference) -> VerificationResult:
            async with semaphore:
                return await self.verify_single_reference(ref)
        
        # Create all verification tasks
        tasks = [verify_with_semaphore(ref) for ref in references]
        
        # Execute tasks and display progress
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                completed += 1
                
                # Display progress
                if completed % 10 == 0 or completed == len(references):
                    progress = (completed / len(references)) * 100
                    logger.info(f"Progress: {completed}/{len(references)} ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Verification task failed: {e}")
                completed += 1
        
        self.stats.end_time = time.time()
        self.stats.total_time = self.stats.end_time - self.stats.start_time
        
        logger.info(f"Batch verification completed, time taken: {self.stats.total_time:.2f}seconds")
        self.stats.print_statistics()
        
        return results
    
    def export_verification_results_to_csv_with_original(self, verification_results: List[VerificationResult], 
                                                        original_references: List[Reference], 
                                                        output_path: str) -> int:
        """
        Export verification results to CSV file (including original citation information)
        """
        if not verification_results:
            logger.warning("No verification results to export")
            return 0
        
        # Create mapping from citation ID to citation
        ref_map = {ref.id: ref for ref in original_references}
        
        # CSV field definitions
        fieldnames = [
            'reference_id', 'final_status', 'diagnosis', 'problematic_fields',
            'original_title', 'found_title', 'original_authors', 'found_authors',
            'original_year', 'found_year', 'found_venue', 'found_url',
            'title_similarity', 'author_similarity', 'verification_time',
            'sources_checked', 'llm_reparsed', 'verification_notes', 'recommendations'
        ]
        
        rows = []
        for result in verification_results:
            # Get original citation
            original_ref = ref_map.get(result.reference_id)
            
            # Extract similarity information and LLM reparse flag
            title_similarity = 0.0
            author_similarity = 0.0
            llm_reparsed = False
            if result.verification_notes:
                for note in result.verification_notes:
                    if "Title similarity:" in note:
                        title_similarity = float(note.split(":")[1].strip())
                    elif "Author similarity:" in note:
                        author_similarity = float(note.split(":")[1].strip())
                    elif "Verification successful after LLM reparse" in note:
                        llm_reparsed = True
            
            # Also check if sources_checked contains llm_reparse
            if result.sources_checked and "llm_reparse" in result.sources_checked:
                llm_reparsed = True
            
            # Prepare row data
            row_data = {
                'reference_id': result.reference_id,
                'final_status': result.final_status.value,
                'diagnosis': result.diagnosis,
                'problematic_fields': ';'.join(result.problematic_fields) if result.problematic_fields else '',
                'original_title': original_ref.title if original_ref else '',
                'found_title': result.best_match.title if result.best_match else '',
                'original_authors': ';'.join(original_ref.authors) if original_ref and original_ref.authors else '',
                'found_authors': ';'.join(result.best_match.authors) if result.best_match and result.best_match.authors else '',
                'original_year': original_ref.year if original_ref else '',
                'found_year': result.best_match.year if result.best_match else '',
                'original_venue': original_ref.venue if original_ref else '',
                'found_venue': result.best_match.venue if result.best_match else '',
                'found_url': result.best_match.url if result.best_match else '',
                'title_similarity': title_similarity,
                'author_similarity': author_similarity,
                'verification_time': f"{result.total_time:.3f}",
                'sources_checked': ';'.join(result.sources_checked) if result.sources_checked else '',
                'llm_reparsed': 'Yes' if llm_reparsed else 'No',
                'verification_notes': ';'.join(result.verification_notes) if result.verification_notes else '',
                'recommendations': ';'.join(result.recommendations) if result.recommendations else ''
            }
            rows.append(row_data)
        
        pd.DataFrame(rows).sort_values(by='reference_id', ascending=True).to_csv(output_path, index=False)
        logger.info(f"Exported {len(verification_results)} verification results to {output_path}")
        return len(verification_results)
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_scholar_statistics()
    
    def print_database_statistics(self):
        """Print database statistics"""
        stats = self.get_database_statistics()
        
        print("\n" + "="*60)
        print("Verification results database statistics")
        print(f"total records: {stats['total_records']}")
        print("="*60)
    
    def print_storage_statistics(self):
        """Print storage statistics"""
        self.storage_service.print_storage_statistics()


class FileProcessor:
    """File processor"""
    
    def __init__(self, verifier: ScrapingDogVerifier):
        self.verifier = verifier
        self.storage_processor = StorageIntegratedFileProcessor(verifier, verifier.storage_service)
    
    async def process_directory(self, dir_path: str, args: argparse.Namespace) -> Dict[str, List[Reference]]:
        """Process files in directory (with storage functionality)"""
        return await self.storage_processor.process_directory_with_storage(dir_path, args)
    
    async def process_single_file(self, file_path: str) -> List[Reference]:
        """Process single file (with storage functionality)"""
        return await self.storage_processor.process_single_file_with_storage(file_path)
    
    async def process_llm_file(self,file_path:str) -> List[Reference]:
        jsons = os.listdir(file_path)
        
        references = {}
        for name in jsons:
            file = os.path.join(file_path,name)
            try:
                with open(file,'r') as jsonfile:
                    data = json.load(jsonfile)
                    if type(data['response'])==type('xx') and data['response']!='':
                        data['response'] = data['response'].replace('null',"''")
                        data['response'] = data['response'].replace(".\n ","")
                        data['response'] = data['response'].replace("```","")
                        data['response'] = data['response'].replace("json\n","")
                        data['response'] = data['response'].replace(".]","]")
                        data = eval(data['response'])
                        print(data)
                        break
                    else: 
                        data = data['response']
                    reflist = data
                references[file] = convert_parsed_references(reflist)
                
            except Exception as e:
                logger.error('Failed to read LLM literature',file,e)
                continue
        return references
    
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


async def start_verify(verifier: ScrapingDogVerifier, references: List[Reference], args: argparse.Namespace) -> List[VerificationResult]:
    """Start verification process"""
    async with verifier:
        verification_results = await verifier.verify_batch(references)
    
    # Export verification results to CSV
    if args.input:
        input_name = Path(args.input).stem
        csv_output = f"{input_name}_verified.csv"
    else:
        csv_output = args.output.replace('.json', '.csv')
    
    dir_main = '/'.join(args.dir.split('/')[:-1])+'/validation_results/'+args.dir.split('/')[-1]
    print(dir_main)
    if args.dir:
        if not os.path.exists(dir_main):
            os.makedirs(dir_main)
        csv_output = os.path.join(dir_main, csv_output)
    
    exported_count = verifier.export_verification_results_to_csv_with_original(
        verification_results, references, csv_output
    )
    print(f"\nExported {exported_count} verification results to {csv_output}")
    print(f"Verification results file: {csv_output}")


def create_sample_references() -> List[Reference]:
    """Create sample reference"""
    sample_data = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": 2017,
            "venue": "NeurIPS",
            "doi": None,
            "pmid": None,
            "isbn": None,
            "patent_number": None,
            "arxiv_id": None,
            "url": None,
            "raw": "[1] A. Vaswani et al., \"Attention Is All You Need,\" NeurIPS, 2017."
        },
    ]
    
    return convert_parsed_references(sample_data)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ScrapingDog batch search program")
    parser.add_argument("--dir", "-d", type=str, help="Input directory path", default=None)
    parser.add_argument("--input", "-i", type=str, help="Input JSON file path")
    parser.add_argument("--inllm", "-l", type=str, help="Input LLM JSON file path")
    parser.add_argument("--output", "-o", type=str, default="scholar_results.json", help="Output JSON file path")
    parser.add_argument("--database", "-db", type=str, default="scholar_results.db", help="Database file path")
    parser.add_argument("--concurrent", "-c", type=int, default=10, help="Maximum concurrency")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--stats-only", action="store_true", help="Only display database statistics")
    parser.add_argument("--clear-db", action="store_true", help="Clear database")
    parser.add_argument("--cleanup-duplicates", action="store_true", help="Clean up duplicate data (keep earliest record)")
    parser.add_argument("--year-filter", type=int, help="Filter export by year")
    parser.add_argument("--dblp-db", type=str, default="dblp_titles.db", help="DBLP SQLite database path")
    parser.add_argument("--dblp-threshold", type=float, default=0.9, help="DBLP title match threshold (ratio)")
    parser.add_argument("--dblp-max-candidates", type=int, default=100000, help="Max DBLP candidates per query")
    parser.add_argument("--disable-dblp", action="store_true", help="Skip DBLP pre-match step")
    
    args = parser.parse_args()
    
    # Create verifier and file processor
    verifier = ScrapingDogVerifier(
        db_path=args.database,
        max_concurrent=args.concurrent,
        dblp_db_path=args.dblp_db,
        dblp_match_threshold=args.dblp_threshold,
        dblp_max_candidates=args.dblp_max_candidates,
        enable_dblp=not args.disable_dblp,
    )
    file_processor = FileProcessor(verifier)
    
    try:
        # Handle special commands
        if args.clear_db:
            verifier.db.clear_all_data()
            print("Database cleared")
            return
        
        if args.cleanup_duplicates:
            print("Cleaning up duplicate data...")
            deleted_count = verifier.db.cleanup_duplicates()
            print(f"Cleanup completed, deleted {deleted_count} duplicate records")
            verifier.print_database_statistics()
            return
        
        if args.stats_only:
            verifier.print_database_statistics()
            return
        
        
        # Load references
        if args.sample:
            print("Using sample reference data")
            references = create_sample_references()
            await start_verify(verifier, references, args)
        elif args.dir:
            print(f"Load references from file directory: {args.dir}")
            references = await file_processor.process_directory(args.dir, args)
            
            for file, ref in tqdm(references.items()):
                args.input = file
                await start_verify(verifier, ref, args)
                print("\n" + "="*60)
        elif args.input:
            print(f"Load references from file: {args.input}")
            references = await file_processor.process_single_file(args.input)
            print(f"Loaded {len(references)} references")
            await start_verify(verifier, references, args)
        elif args.inllm:
            print(f"Load LLM literature from file: {args.inllm}")
            args.dir = args.inllm
            references = await file_processor.process_llm_file(args.inllm)
            print(f"Loaded {len(references)} reference json files")
            for file, ref in tqdm(references.items()):
                print('Process',file)
                args.input = file
                await start_verify(verifier, ref, args)
                print("\n" + "="*60)

        else:
            print("Please specify input file directory for batch processing (--dir) or input file path (--input) or use sample data (--sample)")
            return
        
        # Display database statistics
        verifier.print_database_statistics()
        
        print(f"\nDatabase file: {args.database}")
        print("\nVerification completed!")
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted program")
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        print(f"Program execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

