from typing import Dict, List
import json
import os
import re
import asyncio
from weakref import ref
import aiohttp

from openai import OpenAI
from parser.format.utils import clean_text, extract_id
from parser.utils.pdf_reader import pdf_to_text

MAX_RETRY_TIMES = 3

# LLM API key should be set via environment variable
# export DASHSCOPE_API_KEY='your_api_key_here'
if 'DASHSCOPE_API_KEY' not in os.environ:
    os.environ['DASHSCOPE_API_KEY'] = 'YOUR_API_KEY_HERE'

def llm_parse(text: str) -> List[Dict]:
    """
    Parse text using LLM to extract references
    
    Args:
        text: Input text content
        
    Returns:
        List of parsed references
    """
    return asyncio.run(llm_parse_async(text))

def llm_parse_pdf(pdf_path: str) -> List[Dict]:
    """
    Parse PDF file using LLM to extract references
    
    Args:
        pdf_path: Path to PDF file storage
        :return: List of parsed references
    """
    text = pdf_to_text(pdf_path)
    return llm_parse(text)

async def llm_parse_async(text: str, is_tidy = False) -> List[Dict]:
    """
    Parse text using LLM to extract references
    
    Args:
        text: Input text content
        is_tidy: Whether text is tidied by newline formatting
        
    Returns:
        List of extracted references
    """
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(32) 

    if is_tidy:
        # Split by carriage return and newline
        ref_str_list = [i for i in re.split(r'(\n+)', text) if i.strip()]
    else:
        # Split by []
        ref_str_list = [i for i in re.split(r'(\[[^\]]*\][^\[]*)', text) if i.strip()]

    
    if len(ref_str_list) == 0:
        return []
    
    if len(ref_str_list[-1]) > 256:
        ref_str_list[-1] = ref_str_list[-1][:256]

    ref_list = [{'id': extract_id(ref_str), 'raw': ref_str.strip()} for ref_str in ref_str_list]
    
    tasks = [parse_task(ref_str, ref_list[i], semaphore) for i, ref_str in enumerate(ref_str_list)]
    results = await asyncio.gather(*tasks)
    return results


async def parse_task(text: str, reference: Dict, semaphore: asyncio.Semaphore) -> Dict:
    result = await llm_str2ref(text, semaphore)
    reference.update(result)
    return reference

async def llm_str2ref(raw_str: str, semaphore: asyncio.Semaphore) -> Dict:
    """
        Use large language model to parse text and extract references
        :param text: Input text content
        :return: Dictionary of parsed references
    """

    prompt = f"""
        You are an academic writing assistant that can extract references from academic papers. Please extract references from the following text and output in JSON format:
        {{
            "title": "Title",
            "authors": "Authors",
            "venue": "Journal/Conference/Publication platform name",
            "year": "Year",
            "url": "Link (if available)",
            "volume": "Volume (if available)",
            "number": "Issue (if available)",
            "pages": "Pages (if available)",
            "reference_type": "Reference type, one of the following values: ['article', 'series', 'thesis', 'monograph', 'unknown']",
        }},
        Field description:
        - authors: String array containing all author names
        - Other fields can be omitted if no information is available
        - reference_type field should strictly select from the following types and prioritize the most appropriate type based on reference content:
            - 'article': Conference paper or journal article (journal paper, conference paper, single article in conference proceedings, etc., conference papers should be classified as this type)
            - 'series': Book series, serial publications (e.g., Lecture Notes in Computer Science)
            - 'thesis': Thesis (e.g., PhD, Master's, graduate thesis, etc.)
            - 'monograph': Monograph, book (e.g., a complete book published by an author)
            - 'unknown': Use when type cannot be determined
        The following text is a reference:
        {raw_str}
        Please start extracting the reference:
    """

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts references from academic papers."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    retry_times = MAX_RETRY_TIMES
    while retry_times > 0:
        try: 
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        resp_json = await resp.json()
                        #print(resp_json)
                        raw_reference = resp_json["choices"][0]["message"]["content"].strip()
                        raw_reference = raw_reference[raw_reference.find('{'):raw_reference.rfind('}')+1]
                        reference = {}
                        try:
                            reference = json.loads(raw_reference)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {raw_reference}")
                        #print(f"LLM parsed reference: {reference}")
                        return reference
        except Exception as e:
            if retry_times > 0:
                retry_times -= 1
                print(f"Error occurred: {e}, ref_str: {raw_str}. Retrying... ({MAX_RETRY_TIMES - retry_times}/{MAX_RETRY_TIMES})")
            else:
                print(f"Failed to process reference after retries: {raw_str}")
                return {}
    return {}

# if __name__ == "__main__":
#     sample_text = """
#     [1] Smith, J., & Doe, A. (2020). An overview of machine learning. Journal of AI Research, 45(3), 123-145. https://doi.org/10.1234/jair.2020.5678
#     [2] Brown, B., Green, C., & White, D. (2019). Deep learning techniques for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 678-689.
#     """
#     result = llm_parse(sample_text)
#     print(json.dumps(result, indent=4, ensure_ascii=False))