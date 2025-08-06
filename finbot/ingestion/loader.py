"""
Document Loading Module

Handles loading and text extraction from various document formats
including PDF and HTML files.
"""

import pathlib
from typing import List, Tuple
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup


def load_pdf(file_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as a single string
        
    Raises:
        Exception: If PDF processing fails
    """
    text_pages = []
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
    
    return "\n".join(text_pages)


def load_html(file_path: Path) -> str:
    """
    Extract text content from an HTML file.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        Extracted text content with scripts and styles removed
        
    Raises:
        Exception: If HTML processing fails
    """
    html_content = Path(file_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()
    
    return soup.get_text(separator="\n", strip=True)


def load_sources(directory: str) -> List[Tuple[str, str]]:
    """
    Load all supported documents from a directory.
    
    Args:
        directory: Directory path containing documents
        
    Returns:
        List of tuples containing (file_path, extracted_text)
        
    Supported formats:
        - PDF (.pdf)
        - HTML (.html, .htm)
    """
    directory_path = pathlib.Path(directory)
    document_sources = []
    
    if not directory_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return document_sources
    
    # Process all files in directory recursively
    for file_path in directory_path.glob("**/*"):
        if not file_path.is_file():
            continue
            
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".pdf":
                text_content = load_pdf(file_path)
                document_sources.append((str(file_path), text_content))
                
            elif file_extension in [".html", ".htm"]:
                text_content = load_html(file_path)
                document_sources.append((str(file_path), text_content))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return document_sources