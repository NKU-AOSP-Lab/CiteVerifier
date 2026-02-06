
from calendar import c
from weakref import ref
from grobid_client.grobid_client import GrobidClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from typing import List, Dict
import xml.etree.ElementTree as ET
import os
import re


def grobid_parse(pdf_path: str,xml_save:str) -> List[Dict]:
    """
        Use Grobid to parse PDF files and extract references
        :param pdf_path: Directory path storing PDF files
        :return: List containing dictionary for each reference
    """

    check_temp_path()

    # Copy PDF file to temporary directory
    os.system(f'cp "{pdf_path}" parser/_temp/pdf_input/')
    client = GrobidClient()

    # Process PDF file and generate XML output
    client.process(
        service="processReferences",
        input_path="parser/_temp/pdf_input",
        output="parser/_temp/xml_output",
        include_raw_citations=True,
        n=20
    )

    # Parse all generated XML files
    references = []
    for file in os.listdir("parser/_temp/xml_output"):
        if file.endswith(".xml"):
            xml_path = os.path.join("parser/_temp/xml_output", file)
            references.extend(parse_xml(xml_path))
    
    os.system(f'cp {os.path.join("parser/_temp/xml_output", file)} {xml_save}/{file}')

    clear_temp_path()

    return references

# Currently not feasible
# def grobid_parse_text(text: str) -> List[Dict]:
#     """
#         Use Grobid to parse PDF files and extract references
#         :param pdf_path: Directory path storing PDF files
#         :return: List containing dictionary for each reference
#     """

#     check_temp_path()

#     # Save string as PDF file
#     pdf_path = "parser/_temp/pdf_input/temp.pdf"
#     string_to_pdf(text, pdf_path)
#     # Copy PDF file to temporary directory
#     os.system(f'cp "{pdf_path}" parser/_temp/pdf_input/')

#     # Process PDF file and generate XML output
#     client.process(
#         service="processReferences",
#         input_path="parser/_temp/pdf_input",
#         output="parser/_temp/xml_output",
#         include_raw_citations=True,
#         n=20
#     )

#     # Parse all generated XML files
#     references = []
#     for file in os.listdir("parser/_temp/xml_output"):
#         if file.endswith(".xml"):
#             xml_path = os.path.join("parser/_temp/xml_output", file)
#             references.extend(parse_xml(xml_path))

#     clear_temp_path()

#     return references

def string_to_pdf(text: str, filename: str):
    """
    Save string as PDF file
    :param text: String to save
    :param filename: Output PDF file name
    """
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.drawString(72, height - 72, text)  # Write string at the top of the page
    c.save()


    
def parse_xml(xml_path: str) -> List[Dict]:
    """
    Parse Grobid-generated XML files and extract reference information
    :param xml_path: XML file path
    :return: List containing a dictionary for each reference
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    references = []
    for bibl in root.findall('.//tei:listBibl/tei:biblStruct', ns):
        xml_id = bibl.get('{http://www.w3.org/XML/1998/namespace}id')
        # Numeric ID
        id_match = re.search(r'\d+', xml_id) if xml_id else None
        id = id_match.group(0) if id_match else None
        
        # Grobid IDs start from 0; here we change it to start from 1
        if id is not None:
            id = int(id) + 1

        # Title
        title_elem = bibl.find('.//tei:title[@type="main"]', ns)
        title = title_elem.text if title_elem is not None else None
        title_level = title_elem.get('level') if title_elem is not None else None
        note_elem = bibl.find('.//tei:note', ns)
        note_type = note_elem.get('type') if note_elem is not None else None

        reference_type = get_reference_type(title_level, note_type)

        # Authors
        author_elem = bibl.findall('.//tei:author/tei:persName', ns)
        authors = []
        for author in author_elem:
            forename = author.find('tei:forename', ns)
            surname = author.find('tei:surname', ns)
            full_name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()
            if full_name:
                authors.append(full_name)

        # Publishing platform
        platform_elem = bibl.find('.//tei:monogr/tei:title', ns)
        platform = platform_elem.text if platform_elem is not None else None

        # Publishing year
        date_elem = bibl.find('.//tei:monogr/tei:imprint/tei:date[@type="published"]', ns)
        pub_year = date_elem.text if date_elem is not None else None
        
        year_match = re.search(r'(19\d{2}|20\d{2})', pub_year or '')
        if pub_year and year_match:
            pub_year = year_match.group(0)

        # URL
        url_elem = bibl.find('.//tei:ptr', ns)
        url = url_elem.get('target') if url_elem is not None else None

        # Raw_Reference
        raw_ref_elem = bibl.find('.//tei:note[@type="raw_reference"]', ns)
        raw_reference = raw_ref_elem.text if raw_ref_elem is not None else None

        # volume, issue, pages
        volume_elem = bibl.find('.//tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]', ns)
        volume = volume_elem.text if volume_elem is not None else None

        issue_elem = bibl.find('.//tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]', ns)
        issue = issue_elem.text if issue_elem is not None else None

        pages_elem = bibl.find('.//tei:monogr/tei:imprint/tei:biblScope[@unit="page"]', ns)
        pages = pages_elem.text if pages_elem is not None else None


        references.append(
            {
                "id": id,
                "title": title,
                "authors": authors,
                "venue": platform,
                "year": pub_year,
                "url": url,

                "volume": volume,
                "issue": issue,
                "pages": pages,

                "raw": raw_reference,

                "reference_type": reference_type
            }
        )

    return references

def get_reference_type(title_level, note_type=None):
    if title_level == "a":
        return "article"  # or "chapter"
    elif title_level == "j":
        return "journal"
    elif title_level == "s":
        return "series"
    elif title_level == "m":
        if note_type == "report":
            return "thesis"
        else:
            return "monograph"
    else:
        return "unknown"

def check_temp_path():
    """
    Check and create temporary folder
    """
    if not os.path.exists("parser/_temp/pdf_input"):
        os.makedirs("parser/_temp/pdf_input")
    if not os.path.exists("parser/_temp/xml_output"):
        os.makedirs("parser/_temp/xml_output")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_temp_path():
    """ Clean up temporary folder """
    check_temp_path()
    for folder in ["parser/_temp/pdf_input", "parser/_temp/xml_output"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# if __name__ == "__main__":
    
#     refs = grobid_parse("parser/pdf_example/dnsbomb.pdf")

    

