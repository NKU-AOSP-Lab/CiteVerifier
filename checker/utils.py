"""
Utility functions module
Contains string similarity calculation, data cleaning, author list comparison, and other utilities
"""

import re
import string
import unicodedata
from typing import List, Optional, Tuple, Dict, Any
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)


class StringUtils:
    """String processing utilities class"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text (Unicode normalization, lowercase, remove punctuation)"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean title string"""
        if not title:
            return ""
        
        # Remove bracket content [1], [Online], etc.
        title = re.sub(r'\s*\[.*?\]\s*', ' ', title)
        
        # Remove leading numbers "1. ", "1) ", etc.
        title = re.sub(r'^\s*[\d\.\-\)\]]+\s*', '', title)
        
        # Remove quotes
        title = title.strip('"\'"\'"\'\'"')
        
        # Remove curly braces (but keep content)
        title = re.sub(r'\{([^}]*)\}', r'\1', title)
        
        # Normalize whitespace
        #title = re.sub(r'\s+', ' ', title).strip()

        # Remove hyphens
        title = title.replace('-', '')
        
        prev_title = ""
        while prev_title != title:
            prev_title = title
            # Remove outermost curly braces
            if title.startswith('{') and title.endswith('}'):
                # Check if it's a complete pair of braces
                brace_count = 0
                for i, char in enumerate(title):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    # If brace count is 0 in the middle, it's not a complete outer wrapper
                    if brace_count == 0 and i < len(title) - 1:
                        break
                else:
                    # If loop completes normally, it's a complete outer wrapper, remove it
                    if brace_count == 0:
                        title = title[1:-1].strip()
        return title
    
    @staticmethod
    def extract_main_title(title: str) -> str:
        """Extract main title, remove subtitle part"""
        if not title:
            return ""
        
        # Clean title first
        cleaned = StringUtils.clean_title(title)
        
        # Common subtitle separators: colon, dash, semicolon
        #separators = [':', '—', '–', '-', ';']
        separators = [':']
        
        for sep in separators:
            if sep in cleaned:
                # Split title, take first part as main title
                parts = cleaned.split(sep, 1)
                main_title = parts[0].strip()
                
                # Ensure main title has sufficient length (avoid split errors)
                if len(main_title) >= 10:
                    return main_title
        
        # If no separator found, return original title
        return cleaned
    
    @staticmethod
    def similarity_score(text1: str, text2: str, method: str = "ratio") -> float:
        """
        Calculate similarity score between two strings
        
        Args:
            text1: First string
            text2: Second string
            method: Similarity calculation method ("ratio", "token_sort", "token_set", "partial")
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize text
        text1_norm = StringUtils.normalize_text(text1)
        text2_norm = StringUtils.normalize_text(text2)
        
        if text1_norm == text2_norm:
            return 1.0
        
        if method == "ratio":
            score = fuzz.ratio(text1_norm, text2_norm)
        elif method == "token_sort":
            score = fuzz.token_sort_ratio(text1_norm, text2_norm)
        elif method == "token_set":
            score = fuzz.token_set_ratio(text1_norm, text2_norm)
        elif method == "partial":
            score = fuzz.partial_ratio(text1_norm, text2_norm)
        else:
            # Default to token_sort_ratio
            score = fuzz.token_sort_ratio(text1_norm, text2_norm)
        
        return score / 100.0
    
    @staticmethod
    def enhanced_title_similarity(title1: str, title2: str) -> float:
        """
        Enhanced title similarity calculation that handles subtitles and format variations
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score (0-1)
        """
        if not title1 or not title2:
            return 0.0
        
        # Clean titles (including removing curly braces and format characters)
        clean1 = StringUtils.clean_title(title1)
        clean2 = StringUtils.clean_title(title2)
        
        # Further normalization: lowercase, remove punctuation
        normalized1 = StringUtils._normalize_for_comparison(clean1)
        normalized2 = StringUtils._normalize_for_comparison(clean2)
        
        # 1. Directly compare normalized titles
        direct_similarity = StringUtils.similarity_score(normalized1, normalized2, "ratio")
        
        # If direct similarity is already high, return it
        if direct_similarity >= 0.9:  # Lower threshold, more lenient
            return direct_similarity
        
        # 2. Use maximum score from multiple similarity algorithms
        similarities = [
            StringUtils.similarity_score(normalized1, normalized2, "ratio"),
        ]
        max_similarity = max(similarities)
        
        if max_similarity >= 0.9:  # If any method reaches threshold
            return max_similarity
        
        # 3. Extract main titles for comparison
        main1 = StringUtils.extract_main_title(title1)
        main2 = StringUtils.extract_main_title(title2)

        main1 = ''.join(main1.split(' '))
        main2 = ''.join(main2.split(' '))
        
        if main1 and main2:
            main_norm1 = StringUtils._normalize_for_comparison(main1)
            main_norm2 = StringUtils._normalize_for_comparison(main2)
            
            main_similarities = [
                StringUtils.similarity_score(main_norm1, main_norm2, "ratio"),
            ]
            main_max_similarity = max(main_similarities)
            
            # if main_max_similarity >= 0.9:
            #     # Check if one title contains another title
            #     shorter = normalized1 if len(normalized1) < len(normalized2) else normalized2
            #     longer = normalized2 if len(normalized1) < len(normalized2) else normalized1
                
                # If the shorter title is the beginning of the longer title, consider it a match
                # if longer.startswith(shorter) or StringUtils.similarity_score(shorter, longer, "partial") >= 0.9:
                #     return 0.92  # Assign high similarity
            
            max_similarity = max(max_similarity, main_max_similarity)
        
        return max_similarity
    
    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class AuthorUtils:
    """Author processing utility class"""
    
    @staticmethod
    def normalize_author_name(name: str) -> str:
        """Normalize author name"""
        if not name:
            return ""
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Handle abbreviations
        name = re.sub(r'\.(\s*)', r'\1', name)
        
        # Remove common suffixes
        suffixes = ['Jr.', 'Sr.', 'III', 'II', 'IV', 'Ph.D.', 'M.D.']
        for suffix in suffixes:
            name = re.sub(rf'\s*,?\s*{re.escape(suffix)}\s*$', '', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    @staticmethod
    def extract_last_name(name: str) -> str:
        """
        Extract last name from full name
        """
        # Remove extra whitespace
        name = name.strip()
        # Split name by spaces
        parts = name.split()
        # If space splitting fails such as compound names (e.g., AdamSmith), try splitting by capitals
        if re.match(r'^[A-Z][a-z]+[A-Z][a-z]+$', name):
            parts = re.findall(r'[A-Z][a-z]+', name)
        # If still cannot split, return original name
        if len(parts) == 0:
            return name
        
        surname = name
        for part in parts:
            # If matches [A-Z][a-z]. format, i.e., capitalized abbreviation, skip
            if re.match(r'^[A-Z][a-z]*\.$', part):
                continue
            else:
                surname = part
                # If multiple parts, take the last non-abbreviation part as surname

        return surname
        
    @staticmethod
    def compare_authors(authors1: List[str], authors2: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Compare two author lists
        
        Args:
            authors1: First author list
            authors2: Second author list
            
        Returns:
            (Overall similarity, List of detailed matching results)
        """
        if not authors1 or not authors2:
            return 0.0, []
        
        # Normalize author names
        norm_authors1 = [AuthorUtils.normalize_author_name(a) for a in authors1]
        norm_authors2 = [AuthorUtils.normalize_author_name(a) for a in authors2]
        
        # Calculate best match for each author
        matches = []
        total_score = 0.0
        
        for author1 in norm_authors1:
            if not author1:
                continue
                
            best_match = ""
            best_score = 0.0
            
            for author2 in norm_authors2:
                if not author2:
                    continue
                
                # Calculate full name similarity
                full_score = StringUtils.similarity_score(author1, author2, "token_sort")
                
                # Calculate last name similarity (with higher weight)
                last1 = AuthorUtils.extract_last_name(author1)
                last2 = AuthorUtils.extract_last_name(author2)
                last_score = StringUtils.similarity_score(last1, last2, "ratio")
                
               
                combined_score = last_score*0.5 + full_score*0.5
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = author2
            
            matches.append((author1, best_match, best_score))
            total_score += best_score
        
        # Calculate average similarity
        avg_similarity = total_score / len(norm_authors1) if norm_authors1 else 0.0
        
        return avg_similarity, matches
    
    @staticmethod
    def compare_authors_by_last_name(authors1: List[str], authors2: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Compare two author lists by last name only
        
        Args:
            authors1: First author list
            authors2: Second author list
            
        Returns:
            (Overall similarity, List of detailed matching results)
        """
        if not authors1 or not authors2:
            return 0.0, []
        
        # Extract last names for all authors
        last_names1 = []
        last_names2 = []
        
        for author in authors1:
            if author:
                normalized = AuthorUtils.normalize_author_name(author)
                last_name = AuthorUtils.extract_last_name(normalized)
                if last_name:
                    last_names1.append(last_name.lower())
        
        for author in authors2:
            if author:
                normalized = AuthorUtils.normalize_author_name(author)
                last_name = AuthorUtils.extract_last_name(normalized)
                if last_name:
                    last_names2.append(last_name.lower())
        
        if not last_names1 or not last_names2:
            return 0.0, []
        
        # Calculate best match for each last name
        matches = []
        total_score = 0.0
        
        for last_name1 in last_names1:
            best_match = ""
            best_score = 0.0
            
            for last_name2 in last_names2:
                # Direct comparison of last names
                if last_name1 == last_name2:
                    score = 1.0  # Exact match
                else:
                    # Calculate last name similarity
                    score = StringUtils.similarity_score(last_name1, last_name2, "ratio")
                
                if score > best_score:
                    best_score = score
                    best_match = last_name2
            
            matches.append((last_name1, best_match, best_score))
            total_score += best_score
        
        # Calculate average similarity
        avg_similarity = total_score / len(last_names1) if last_names1 else 0.0
        
        return avg_similarity, matches
    
    @staticmethod
    def get_first_author(authors: List[str]) -> Optional[str]:
        """Get first author"""
        return authors[0] if authors else None


class DataUtils:
    """Data processing utility class"""
    
    @staticmethod
    def safe_int(value: Any) -> Optional[int]:
        """Safely convert to integer"""
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                # Extract digits
                match = re.search(r'\d+', value)
                if match:
                    return int(match.group())
            return int(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def extract_year(text: str) -> Optional[int]:
        """Extract year from text"""
        if not text:
            return None
        
        # Find 4-digit year (1900-2100)
        year_pattern = r'\b(19|20)\d{2}\b'
        matches = re.findall(year_pattern, text)
        
        if matches:
            years = [int(f"{match}") for match in matches]
            # Return the most reasonable year closest to current year
            current_year = 2025
            valid_years = [y for y in years if 1900 <= y <= current_year + 5]
            if valid_years:
                return max(valid_years)
        
        return None
    
    @staticmethod
    def clean_doi(doi: str) -> Optional[str]:
        """Clean DOI string"""
        if not doi:
            return None
        
        # Remove prefix
        doi = re.sub(r'^(doi:\s*|https?://dx\.doi\.org/|https?://doi\.org/)', '', doi, flags=re.IGNORECASE)
        
        # DOI format validation (basic check)
        if re.match(r'^10\.\d{4,}/.+', doi):
            return doi
        
        return None
    
    @staticmethod
    def clean_isbn(isbn: str) -> Optional[str]:
        """Clean ISBN string"""
        if not isbn:
            return None
        
        # Remove all non-digit and non-X characters
        isbn = re.sub(r'[^\dX]', '', isbn.upper())
        
        # ISBN-10 or ISBN-13 validation
        if len(isbn) == 10:
            return isbn
        elif len(isbn) == 13:
            return isbn
        
        return None
    
    @staticmethod
    def clean_patent_number(patent_number: str) -> Optional[str]:
        """Clean patent number string"""
        if not patent_number:
            return None
        
        # Remove extra whitespace and punctuation
        patent_number = re.sub(r'\s+', '', patent_number)
        patent_number = re.sub(r'[,\.]', '', patent_number)
        
        # Basic patent number format check
        # US patents: US1234567B2, US20210123456A1
        # Other country patent formats can be added as needed
        if re.match(r'^[A-Z]{2}\d+[A-Z]\d*$', patent_number):
            return patent_number
        
        return patent_number  # Return cleaned version even if format is non-standard
    
    @staticmethod
    def normalize_venue(venue: str) -> str:
        """Normalize journal/conference names"""
        if not venue:
            return ""
        
        # Remove common words
        common_words = [
            'journal', 'of', 'the', 'proceedings', 'conference', 
            'international', 'annual', 'symposium', 'workshop'
        ]
        
        venue_norm = StringUtils.normalize_text(venue)
        words = venue_norm.split()
        
        # Keep important words
        important_words = [w for w in words if w not in common_words or len(w) > 3]
        
        return ' '.join(important_words)
    
    @staticmethod
    def extract_page_numbers(pages: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract start and end page numbers from page string"""
        if not pages:
            return None, None
        
        # Match various page format: "123-456", "123–456", "123-56", "123"
        page_pattern = r'(\d+)(?:[-–](\d+))?'
        match = re.search(page_pattern, pages)
        
        if match:
            start_page = int(match.group(1))
            end_page_str = match.group(2)
            
            if end_page_str:
                end_page = int(end_page_str)
                # Handle abbreviated form "123-56" -> "123-156"
                if end_page < start_page and len(end_page_str) < len(str(start_page)):
                    start_str = str(start_page)
                    end_page = int(start_str[:-len(end_page_str)] + end_page_str)
            else:
                end_page = start_page
            
            return start_page, end_page
        
        return None, None


class ValidationUtils:
    """Validation utility class"""
    
    @staticmethod
    def is_valid_year(year: Optional[int]) -> bool:
        """Validate if year is reasonable"""
        if year is None:
            return False
        return 1900 <= year <= 2030
    
    @staticmethod
    def is_valid_doi(doi: Optional[str]) -> bool:
        """Validate DOI format"""
        if not doi:
            return False
        
        clean_doi = DataUtils.clean_doi(doi)
        return clean_doi is not None
    
    @staticmethod
    def calculate_confidence_level(similarity_score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity_score >= 0.9:
            return "high"
        elif similarity_score >= 0.7:
            return "medium"
        elif similarity_score >= 0.5:
            return "low"
        else:
            return "very_low"


class PerformanceUtils:
    """Performance monitoring utility class"""
    
    @staticmethod
    def measure_time(func):
        """Decorator: Measure function execution time"""
        import time
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                logger.debug(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                logger.debug(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        
        # Check if it's an async function
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Convenience functions
def calculate_overall_similarity(field_similarities: Dict[str, float], field_weights: Dict[str, float]) -> float:
    """
    Calculate overall similarity score
    
    Args:
        field_similarities: Similarity scores for each field
        field_weights: Weights for each field
        
    Returns:
        Weighted average similarity score
    """
    total_weight = 0.0
    weighted_sum = 0.0
    
    for field, similarity in field_similarities.items():
        weight = field_weights.get(field, 0.1)
        weighted_sum += similarity * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def quick_reference_comparison(ref1: Dict[str, Any], ref2: Dict[str, Any]) -> Dict[str, float]:
    """
    Quick comparison of two references
    
    Args:
        ref1: First reference dictionary
        ref2: Second reference dictionary
        
    Returns:
        Dictionary of similarity scores for each field
    """
    similarities = {}
    
    # Compare titles
    if ref1.get('title') and ref2.get('title'):
        title1 = StringUtils.clean_title(ref1['title'])
        title2 = StringUtils.clean_title(ref2['title'])
        similarities['title'] = StringUtils.similarity_score(title1, title2)
    
    # Compare authors
    if ref1.get('authors') and ref2.get('authors'):
        auth_sim, _ = AuthorUtils.compare_authors(ref1['authors'], ref2['authors'])
        similarities['authors'] = auth_sim
    
    # Compare years
    if ref1.get('year') and ref2.get('year'):
        similarities['year'] = 1.0 if ref1['year'] == ref2['year'] else 0.0
    
    # Compare journal/conference
    if ref1.get('venue') and ref2.get('venue'):
        venue1 = DataUtils.normalize_venue(ref1['venue'])
        venue2 = DataUtils.normalize_venue(ref2['venue'])
        similarities['venue'] = StringUtils.similarity_score(venue1, venue2)
    
    # Compare DOI
    if ref1.get('doi') and ref2.get('doi'):
        doi1 = DataUtils.clean_doi(ref1['doi'])
        doi2 = DataUtils.clean_doi(ref2['doi'])
        if doi1 and doi2:
            similarities['doi'] = 1.0 if doi1 == doi2 else 0.0
    
    return similarities

# Example usage:
# s1 = 'The effect of information visualization delivery on narrative construction and user experience'
# s2 = 'The Effect of Information Visualization Delivery on Narrative Construction and Development'
# print(StringUtils.enhanced_title_similarity(s1, s2))