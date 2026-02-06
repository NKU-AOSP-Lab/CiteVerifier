"""
Google Search API Client
Uses ScrapingDog API to perform Google search
"""

import logging
from typing import Optional, List, Dict, Any
from urllib.parse import quote

from .base_client import BaseApiClient
from checker.models import Reference, ExternalReference, FieldMatchResult
from checker.config import config
from checker.utils import StringUtils, AuthorUtils, DataUtils

logger = logging.getLogger(__name__)


class GoogleSearchClient(BaseApiClient):
    """Google Search API Client"""
    
    def __init__(self):
        import os
        from checker.config import ApiConfig
        api_key = os.getenv('SCRAPINGDOG_API_KEY', 'YOUR_API_KEY_HERE')
        api_config = ApiConfig(
            base_url="https://api.scrapingdog.com",
            api_key=api_key,
            timeout=5,
            headers={
                "Accept": "application/json",
                "User-Agent": "ReferenceVerificationBot/1.0"
            },
            rate_limit=1.0
        )
        super().__init__(api_config, "google_search")
    
    async def search_reference(self, reference: Reference) -> ExternalReference:
        """
        Search for a reference in Google Search
        
        Args:
            reference: Input reference object
            
        Returns:
            External reference information from data sources
            
        Raises:
            Exception: Raised when search fails
        """
        # Build search query
        query = reference.title
        if not query:
            raise ValueError("Insufficient information for Google search")
        
        # Execute search
        return await self._search_by_query(query, reference)
    
    async def _search_by_query(self, query: str, reference: Reference) -> ExternalReference:
        """Search by query string"""
        url = f"{self.config.base_url}/google"
        params = {
            "api_key": self.config.api_key,
            "results": 10,
            "country": "us",
            "advance_search": "false",
            "domain": "google.com",
            "language": "en",
            "page": 0,
            "query": query,
        }
        
        logger.debug(f"Searching Google with query: {query}")
        
        response = await self.make_request("GET", url, params=params)
        
        if 'organic_results' not in response or not response['organic_results']:
            raise ValueError("No search results found")

        # Find best matching result (threshold 0.9)
        best_result = self._find_best_match_with_threshold(response['organic_results'], reference)
        return self._parse_google_result(best_result, query)
    
    def _find_best_match_with_threshold(self, results: List[Dict], reference: Reference, threshold: float = 0.9) -> Dict:
        """
        Find the best matching result from search results.
        If a result with title similarity >= threshold exists, return the first one.
        Otherwise return the result with highest similarity.
        """
        best_score = 0.0
        best_result = results[0]

        for result in results:
            # Title matching score (using enhanced_title_similarity)
            if result.get('title') and reference.title:
                title_sim = StringUtils.enhanced_title_similarity(
                    reference.title,
                    result['title']
                )

                # If found result with similarity >= threshold, return directly
                if title_sim >= threshold:
                    logger.info(f"Found high similarity match (similarity: {title_sim:.2f}): {result.get('title', '')[:50]}...")
                    return result

                # Record highest score result
                if title_sim > best_score:
                    best_score = title_sim
                    best_result = result

        # Did not find result meeting threshold, return highest similarity result
        logger.info(f"No results with similarity >= {threshold} found, returning highest similarity result (similarity: {best_score:.2f})")
        return best_result

    def _find_best_match(self, results: List[Dict], reference: Reference) -> Dict:
        """Find best matching result from search results (deprecated method)"""
        best_score = 0.0
        best_result = results[0]

        for result in results:
            score = 0.0

            # Title matching score
            if result.get('title') and reference.title:
                title_sim = StringUtils.similarity_score(
                    StringUtils.clean_title(reference.title),
                    StringUtils.clean_title(result['title'])
                )
                score += title_sim * 0.7

            # Link relevance score (academic websites prioritized)
            link = result.get('link', '').lower()
            if any(domain in link for domain in ['scholar.google', 'arxiv.org', 'ieee.org', 'acm.org', 'springer.com', 'sciencedirect.com']):
                score += 0.2

            # Abstract/description matching score
            if result.get('snippet') and reference.title:
                snippet_sim = StringUtils.similarity_score(
                    StringUtils.clean_title(reference.title),
                    StringUtils.clean_title(result['snippet'])
                )
                score += snippet_sim * 0.1

            if score > best_score:
                best_score = score
                best_result = result

        return best_result
    
    def _parse_google_result(self, result: Dict[str, Any], original_query: str) -> ExternalReference:
        """Parse Google search result"""
        # Extract title
        title = result.get('title')
        title = title.replace('\xa0', ' ')
        if ' - ' in title:
            title = title.split(' - ',1)[0]
        
        # Extract link
        url = result.get('displayed_link')
        url = url.replace('\xa0', ' ')
        if ' › ' in url:
            url = url.split(' › ',1)[0]
        
        return ExternalReference(
            title=title,
            authors=None,
            year=None,
            url=url,
            abstract=None,
            source="google_search",
            metadata={
                'search_engine': 'google'
            }
        )