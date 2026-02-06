"""
Data structure definition module
"""

import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class VerificationStatus(str, Enum):
    """Verification status enum"""
    VALID = "valid"                 # verification passed
    INVALID = "invalid"             # verification failed (fake/error)
    SUSPICIOUS = "suspicious"       # suspicious (partial information error)
    UNVERIFIED = "unverified"       # cannot verify (insufficient information)
    ERROR = "error"                 # error during verification
    TIMEOUT = "timeout"             # verification timeout


class ReferenceType(str, Enum):
    """Reference type enum"""
    JOURNAL_ARTICLE = "journal_article"     # journal article
    CONFERENCE_PAPER = "conference_paper"   # conference paper
    BOOK = "book"                          # book
    BOOK_CHAPTER = "book_chapter"          # book chapter
    PATENT = "patent"                      # patent
    TECHNICAL_REPORT = "technical_report"   # technical report
    THESIS = "thesis"                      # thesis
    PREPRINT = "preprint"                  # preprint
    WEBSITE = "website"                    # website
    OTHER = "other"                        # other


class Reference(BaseModel):
    """Input reference data structure"""
    id: int = Field(..., description="Literature ID")
    title: str = Field(..., description="Literature title")
    authors: List[str] = Field(..., description="Author list")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Published journal/conference/publisher")
    volume: Optional[str] = Field(None, description="Volume")
    issue: Optional[str] = Field(None, description="Issue")
    pages: Optional[str] = Field(None, description="Pages")
    doi: Optional[str] = Field(None, description="DOI identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    isbn: Optional[str] = Field(None, description="ISBN (book)")
    patent_number: Optional[str] = Field(None, description="Patent number")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID")
    url: Optional[str] = Field(None, description="URL link")
    reference_type: Optional[str] = Field(None, description="Reference type")
    raw: Optional[str]  = Field(..., description="Original reference string")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Deep Learning",
                "authors": ["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"],
                "year": 2015,
                "venue": "Nature",
                "volume": "521",
                "issue": "7553",
                "pages": "436-444",
                "doi": "10.1038/nature14539",
                "raw": "[1] Y. LeCun, Y. Bengio, and G. Hinton, \"Deep learning,\" Nature, vol. 521, no. 7553, pp. 436–444, 2015."
            }
        }


class ExternalReference(BaseModel):
    """Literature information returned from external data sources"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    source: str = Field(..., description="Data source identifier (crossref, pubmed, arxiv)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class FieldMatchResult(BaseModel):
    """Single field match result"""
    field_name: str = Field(..., description="Field name")
    input_value: Optional[str] = Field(None, description="Input value")
    external_value: Optional[str] = Field(None, description="External source value")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    is_match: bool = Field(..., description="Whether matched")


class ApiResult(BaseModel):
    """Verification result from single API client"""
    source: str = Field(..., description="API data source name")
    status: VerificationStatus = Field(..., description="Verification status")
    external_ref: Optional[ExternalReference] = Field(None, description="Found external literature information")
    field_matches: List[FieldMatchResult] = Field(default_factory=list, description="Field match details")
    overall_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Overall similarity score")
    error_message: Optional[str] = Field(None, description="Error message")
    response_time: float = Field(0.0, description="API response time(seconds)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Verification timestamp")


class VerificationResult(BaseModel):
    """Final verification result"""
    reference_id: int = Field(..., description="Reference ID")
    final_status: VerificationStatus = Field(..., description="Final verification status")
    overall_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Overall similarity score")
    
    # Diagnostic information
    diagnosis: str = Field(..., description="Detailed diagnostic information")
    problematic_fields: List[str] = Field(default_factory=list, description="Problematic fields")
    
    # Detailed results from each API
    api_results: List[ApiResult] = Field(default_factory=list, description="Verification results from each API")
    
    # Summary information
    best_match: Optional[ExternalReference] = Field(None, description="Best matching external literature")
    sources_checked: List[str] = Field(default_factory=list, description="Data sources checked")
    sources_failed: List[str] = Field(default_factory=list, description="Failed data sources")
    
    # Performance statistics
    total_time: float = Field(0.0, description="Total verification time(seconds)")
    fastest_source: Optional[str] = Field(None, description="Fastest responding data source")
    
    # Detailed explanation
    verification_notes: List[str] = Field(default_factory=list, description="Verification notes")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Verification completion time")

    class Config:
        json_schema_extra = {
            "example": {
                "reference_id": 1,
                "final_status": "valid",
                "overall_similarity": 0.95,
                "diagnosis": "Verified through DOI",
                "sources_checked": ["crossref", "pubmed"],
                "total_time": 2.5,
                "verification_notes": ["Title and authors match exactly", "DOI confirmed"],
                "recommendations": ["Reference appears to be accurate"]
            }
        }

def clean_title(title: str) -> str:
        """Clean title string, remove multilevel braces and other format characters"""
        if not title:
            return ""
        
        title = title.strip()
        
        # Recursively remove braces, handle multilevel nesting
        # For example: {A Formal Analysis of SCTP: Attack Synthesis and Patch Verification}
        # or: {{Title}} -> {Title} -> Title
        prev_title = ""
        while prev_title != title:
            prev_title = title
            # Remove outermost braces
            if title.startswith('{') and title.endswith('}'):
                # Check if it's a complete brace pair
                brace_count = 0
                for i, char in enumerate(title):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    # If brace count is 0 at middle position, it's not a complete outer wrapping
                    if brace_count == 0 and i < len(title) - 1:
                        break
                else:
                    # If loop completes normally, it's a complete outer wrapping and can be removed
                    if brace_count == 0:
                        title = title[1:-1].strip()
        
        # Remove other common format characters
        # Remove bracket content [1], [Online], etc.
        title = re.sub(r'\s*\[.*?\]\s*', ' ', title)
        
        # Remove leading numbers "1. ", "1) ", etc.
        title = re.sub(r'^\s*[\d\.\-\)\]]+\s*', '', title)
        
        # Remove quotes
        title = title.strip('"\'""''')

        # Remove hyphens from line breaks-
        title = title.replace('-', '')
        
        # Normalize whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title

def convert_parsed_reference(parsed_ref_dict: Dict[str, Any], ref_id: int) -> Reference:
    """
    Convert single dictionary returned by parser to Reference object
    
    Args:
        parsed_ref_dict: Reference dictionary parsed by parser
        ref_id: Reference ID
        
    Returns:
        Reference object
    """
    # Process year field
    year = parsed_ref_dict.get('year')
    if year is not None:
        if isinstance(year, str):
            # Extract numbers from string
            import re
            year_match = re.search(r'\d{4}', str(year))
            if year_match:
                try:
                    year = int(year_match.group())
                except (ValueError, TypeError):
                    year = None
            else:
                year = None
        elif not isinstance(year, int):
            year = None
    
    # title, authors, raw fields need default values
    title = parsed_ref_dict.get('title', '')
    if not title or title.strip() == '':
        title = 'Untitled'
    #title = clean_title(title)
    


    if parsed_ref_dict.get('doi') is not None:
        if "\\" in parsed_ref_dict.get('doi'):
            parsed_ref_dict['doi'] = parsed_ref_dict['doi'].replace('\\', '')
    
    authors = parsed_ref_dict.get('authors', [])
    if not isinstance(authors, list):
        if isinstance(authors, str) and authors.strip():
            authors = [authors.strip()]
        else:
            authors = []
    
    raw = parsed_ref_dict.get('raw', '')
    if not raw or raw.strip() == '':
        raw = f"Reference {ref_id}"
    
    return Reference(
        id=ref_id,
        title=title,
        authors=authors,
        year=year,
        venue=parsed_ref_dict.get('venue'),
        volume=parsed_ref_dict.get('volume'),
        issue=parsed_ref_dict.get('issue'),
        pages=parsed_ref_dict.get('pages'),
        doi=parsed_ref_dict.get('doi'),
        pmid=parsed_ref_dict.get('pmid'),
        isbn=parsed_ref_dict.get('isbn'),
        patent_number=parsed_ref_dict.get('patent_number'),
        arxiv_id=parsed_ref_dict.get('arxiv_id'),
        url=parsed_ref_dict.get('url'),
        reference_type=parsed_ref_dict.get('reference_type'),
        raw=raw
    )


def convert_parsed_references(parsed_references: List[Dict[str, Any]]) -> List[Reference]:
    """
    Convert dictionary list returned by parser to Reference object list
    
    Args:
        parsed_references: List of reference dictionaries parsed by parser
        
    Returns:
        List of Reference objects
    """
    #print(parsed_references)
    references = []
    for i, parsed_ref in enumerate(parsed_references):
        # If id already exists in dictionary, use it; otherwise use index+1
        ref_id = parsed_ref.get('id', i + 1)
        
        # Handle None values and string conversion
        if ref_id is None:
            ref_id = i + 1
        elif isinstance(ref_id, str):
            try:
                ref_id = int(ref_id)
            except ValueError:
                ref_id = i + 1
        elif not isinstance(ref_id, int):
            ref_id = i + 1
        
        ref_obj = convert_parsed_reference(parsed_ref, ref_id)
        references.append(ref_obj)
    
    return references


