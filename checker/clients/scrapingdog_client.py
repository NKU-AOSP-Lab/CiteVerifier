"""
ScrapingDog Google Scholar API Client
Used to verify academic literature through Google Scholar search
"""

import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import quote

from .base_client import BaseApiClient
from checker.models import Reference, ExternalReference, FieldMatchResult
from checker.config import config
from checker.utils import StringUtils, AuthorUtils, DataUtils

logger = logging.getLogger(__name__)


class ScrapingDogClient(BaseApiClient):
    """ScrapingDog Google Scholar API Client"""
    
    def __init__(self):
        api_config = config.get_api_config("scrapingdog")
        super().__init__(api_config, "scrapingdog")
        
        # Year matching pattern
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    
    async def search_reference(self, reference: Reference) -> ExternalReference:
        """
        Search for references in Google Scholar
        
        Args:
            reference: Input reference
            
        Returns:
            Literature information from external data sources
            
        Raises:
            Exception: Raised when search fails
        """
        # Build search query
        query = self._build_search_query(reference)
        if not query:
            raise ValueError("Insufficient information for ScrapingDog search")
        
        # Execute search
        return await self._search_google_scholar(query)
    
    def _build_search_query(self, reference: Reference) -> str:
        """Build search query string"""
        query_parts = []
        
        # Prioritize using title for search
        if reference.title:
            clean_title = StringUtils.clean_title(reference.title)
            if clean_title:
                # Use full title or keywords
                # if len(clean_title.split()) > 8:
                #     # Title too long, select keywords
                #     title_words = clean_title.split()
                #     important_words = [word for word in title_words[:6] if len(word) > 3]
                #     if important_words:
                #         query_parts.append(' '.join(important_words))
                #     else:
                #         query_parts.append(clean_title)
                # else:
                query_parts.append(clean_title)
        
        # Add author information
        # if reference.authors:
        #     first_author = AuthorUtils.get_first_author(reference.authors)
        #     if first_author:
        #         last_name = AuthorUtils.extract_last_name(first_author)
        #         if last_name and len(last_name) > 2:
        #             query_parts.append(last_name)
        
        # # Add year information
        # if reference.year:
        #     query_parts.append(str(reference.year))
        
        return ' '.join(query_parts)
    
    async def _search_google_scholar(self, query: str, max_results: int = 10) -> ExternalReference:
        """Search in Google Scholar"""
        search_url = self.config.base_url
        params = {
            "api_key": self.config.api_key,
            "query": query,
            "language": "en",
            "page": 0,
            "results": 10
        }
        
        logger.debug(f"Searching Google Scholar with query: {query}")
        
        response = await self.make_request("GET", search_url, params=params)

        if 'scholar_results' not in response or not response['scholar_results'] or len(response['scholar_results']) == 0:
            raise ValueError("No matching papers found in Google Scholar")

        # Find best matching result (threshold 0.9)
        best_result = self._find_best_match_with_threshold(response['scholar_results'], query)
        return self._parse_scholar_result(best_result)

    def _find_best_match_with_threshold(self, results: List[Dict], query: str, threshold: float = 0.9) -> Dict:
        """
        Find the best matching result from search results.
        If a result with title similarity >= threshold exists, return the first one that meets the condition.
        Otherwise return the result with the highest similarity.

        Args:
            results: List of Google Scholar search results
            query: Original query string (usually the title of the reference)
            threshold: Similarity threshold, default 0.9

        Returns:
            Best matching result
        """
        best_score = 0.0
        best_result = results[0]

        for result in results:
            # Title matching score (using enhanced_title_similarity)
            result_title = result.get('title', '').strip()
            if result_title and query:
                title_sim = StringUtils.enhanced_title_similarity(
                    query,
                    result_title
                )

                # If found result with similarity >= threshold, return directly
                if title_sim >= threshold:
                    logger.info(f"Found high similarity match (similarity: {title_sim:.2f}): {result_title[:50]}...")
                    return result

                # Record highest score result
                if title_sim > best_score:
                    best_score = title_sim
                    best_result = result

        # Did not find result meeting threshold, return highest similarity result
        logger.info(f"No results with similarity >= {threshold} found, returning highest similarity result (similarity: {best_score:.2f})")
        return best_result

    def _parse_scholar_result(self, result: Dict[str, Any]) -> ExternalReference:
        """Parse Google Scholar search results"""
        authors = []
        venue = None
        year = None
        url = None
        try:
            # Extract title
            title = result.get('title', '').strip()
            title = StringUtils.clean_title(title)
            if not title:
                raise ValueError("No title found in Google Scholar result")
            # Extract information
            displayed_link = result.get('displayed_link', '')
            displayed_link = displayed_link.replace('\xa0', ' ')
          
            if displayed_link and ' - ' in displayed_link:
                # Try to extract author information from displayed_link
                # Format is usually: "authors - venue - year"
                parts = displayed_link.split(' - ',1)
                if ',' in parts[0]:
                    authors = parts[0].split(',')
                else:
                    authors = [parts[0]]
                   
               
                
                year_match = self.year_pattern.search(parts[1])
                
                if year_match:
                    try:
                        year = int(year_match.group())
                    except ValueError:
                        pass
            
        
            
                
                if len(parts)>2:
                    venue = parts[1].split(',',1)[0]
                
            # Extract URL
            url = result.get('title_link', '')
            
            # Extract abstract
            #abstract = result.get('snippet', '').strip()
            
            return ExternalReference(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                #abstract=abstract,
                source="scrapingdog",
                metadata={
                    'displayed_link': displayed_link,
                    'result_id': result.get('result_id'),
                    'position': result.get('position'),
                    'search_engine': 'google_scholar'
                }
            )
            
        except Exception as e:
            logger.warning(f"Error parsing ScrapingDog result: {e}")
            raise ValueError(f"Failed to parse search result: {e}")
    
    def calculate_similarity(self, input_ref: Reference, external_ref: ExternalReference) -> float:
        """
        Calculate the similarity between input reference and external reference
        
        Args:
            input_ref: Input reference
            external_ref: Literature information returned from external data sources
            
        Returns:
            Similarity score (0-1)
        """
        if not external_ref:
            return 0.0
        
        similarities = {}
        weights = config.SIMILARITY_CONFIG["field_weights"]
        
        # Title similarity
        if input_ref.title and external_ref.title:
            similarities["title"] = StringUtils.enhanced_title_similarity(
                input_ref.title, external_ref.title
            )
        
        # Author similarity
        if input_ref.authors and external_ref.authors:
            author_sim, _ = AuthorUtils.compare_authors_by_last_name(
                input_ref.authors, external_ref.authors
            )
            similarities["authors"] = author_sim
        
        # Year matching
        if input_ref.year and external_ref.year:
            similarities["year"] = 1.0 if input_ref.year == external_ref.year else 0.0
        
        # Venue similarity
        if input_ref.venue and external_ref.venue:
            similarities["venue"] = StringUtils.fuzzy_match(
                input_ref.venue, external_ref.venue
            )
        
        # Calculate weighted average similarity
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field, similarity in similarities.items():
            weight = weights.get(field, 0.1)
            weighted_sum += similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _generate_field_matches(self, input_ref: Reference, external_ref: ExternalReference) -> List[FieldMatchResult]:
        """Generate field match details"""
        matches = []
        
        # Title match
        if input_ref.title and external_ref.title:
            similarity = StringUtils.enhanced_title_similarity(
                input_ref.title, external_ref.title
            )
            matches.append(FieldMatchResult(
                field_name="title",
                input_value=input_ref.title,
                external_value=external_ref.title,
                similarity_score=similarity,
                is_match=similarity >= config.get_similarity_threshold("title")
            ))
        
        # Author match
        if input_ref.authors and external_ref.authors:
            author_sim, _ = AuthorUtils.compare_authors_by_last_name(
                input_ref.authors, external_ref.authors
            )
            matches.append(FieldMatchResult(
                field_name="authors",
                input_value=", ".join(input_ref.authors),
                external_value=", ".join(external_ref.authors),
                similarity_score=author_sim,
                is_match=author_sim >= config.get_similarity_threshold("authors")
            ))
        
        # Year match
        if input_ref.year and external_ref.year:
            year_match = input_ref.year == external_ref.year
            matches.append(FieldMatchResult(
                field_name="year",
                input_value=str(input_ref.year),
                external_value=str(external_ref.year),
                similarity_score=1.0 if year_match else 0.0,
                is_match=year_match
            ))
        
        # Venue match
        if input_ref.venue and external_ref.venue:
            venue_sim = StringUtils.fuzzy_match(input_ref.venue, external_ref.venue)
            matches.append(FieldMatchResult(
                field_name="venue",
                input_value=input_ref.venue,
                external_value=external_ref.venue,
                similarity_score=venue_sim,
                is_match=venue_sim >= config.get_similarity_threshold("venue")
            ))
        
        return matches