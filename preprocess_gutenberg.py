"""
Gutenberg Book Preprocessor
Strips headers/footers and cleans text from Project Gutenberg books.
"""

import re
from pathlib import Path


def strip_gutenberg_metadata(text: str) -> str:
    """Remove Project Gutenberg header and footer from text."""
    
    # Pattern for start marker
    start_patterns = [
        r'\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*START OF THIS PROJECT GUTENBERG.*?\*\*\*',
        r'Produced by.*?\n\n',
    ]
    
    # Pattern for end marker
    end_patterns = [
        r'\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*END OF THIS PROJECT GUTENBERG.*?\*\*\*',
        r'End of the Project Gutenberg.*',
        r'End of Project Gutenberg.*',
    ]
    
    # Find start
    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            start_idx = match.end()
            break
    
    # Find end
    end_idx = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text[start_idx:], re.IGNORECASE | re.DOTALL)
        if match:
            end_idx = start_idx + match.start()
            break
    
    # Additional cleanup: remove illustration markers and publisher info
    cleaned = text[start_idx:end_idx].strip()
    
    # Remove [Illustration: ...] blocks
    cleaned = re.sub(r'\[Illustration:?[^\]]*\]', '', cleaned)
    
    # Remove publisher/printer info blocks
    cleaned = re.sub(r'CHISWICK PRESS.*?LONDON\.', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'GEORGE ALLEN.*?LONDON', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'RUSKIN HOUSE.*?LONDON', '', cleaned, flags=re.DOTALL)
    
    # Remove TOC entries (lines with dots leading to page numbers)
    cleaned = re.sub(r'^[A-Z][a-z].*\.\.\.\s*\d+$', '', cleaned, flags=re.MULTILINE)
    
    return cleaned.strip()


def clean_special_characters(text: str) -> str:
    """Clean and normalize special characters."""
    
    # Replace curly quotes with straight quotes
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201C': '"',   # Left double quote
        '\u201D': '"',   # Right double quote
        '\u2014': '--',  # Em dash
        '\u2013': '-',   # En dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\ufb01': 'fi',  # fi ligature
        '\ufb02': 'fl',  # fl ligature
        '\r\n': '\n',    # Windows line endings
        '\r': '\n',      # Old Mac line endings
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def split_into_chapters(text: str) -> list:
    """Split text into chapters with metadata."""
    
    # Common chapter patterns
    chapter_patterns = [
        r'^(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)[\.\s]*(.*?)$',
        r'^(?:BOOK|Book)\s+([IVXLCDM]+|\d+)[\.\s]*(.*?)$',
        r'^([IVXLCDM]+)\.\s*(.*?)$',
    ]
    
    # Combine patterns
    combined_pattern = '|'.join(f'({p})' for p in chapter_patterns)
    
    # Find all chapter markers
    chapters = []
    matches = list(re.finditer(combined_pattern, text, re.MULTILINE))
    
    if not matches:
        # Return whole text as single chapter
        return [{'number': 1, 'title': 'Full Text', 'text': text}]
    
    for i, match in enumerate(matches):
        # Extract chapter number and title
        groups = match.groups()
        chapter_num = next((g for g in groups[:3] if g), str(i + 1))
        chapter_title = next((g for g in groups[3:6] if g), '')
        
        # Extract chapter text
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()
        
        chapters.append({
            'number': chapter_num,
            'title': chapter_title.strip(),
            'text': chapter_text
        })
    
    return chapters


def preprocess_gutenberg_book(input_path: str, output_path: str = None):
    """
    Complete preprocessing pipeline for a Gutenberg book.
    
    Args:
        input_path: Path to raw Gutenberg text file
        output_path: Path to save cleaned text (optional)
    
    Returns:
        Cleaned text
    """
    # Read raw text
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"Raw text: {len(raw_text)} chars")
    
    # Strip metadata
    clean_text = strip_gutenberg_metadata(raw_text)
    print(f"After stripping metadata: {len(clean_text)} chars")
    
    # Clean special characters
    clean_text = clean_special_characters(clean_text)
    print(f"After cleaning special chars: {len(clean_text)} chars")
    
    # Split into chapters
    chapters = split_into_chapters(clean_text)
    print(f"Found {len(chapters)} chapters")
    
    # Save if output path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        print(f"Saved to: {output_path}")
    
    return clean_text


if __name__ == '__main__':
    # Process Pride and Prejudice
    input_path = r'C:\projects\books\pride_and_prejudice.txt'
    output_path = r'C:\projects\books\pride_and_prejudice_clean.txt'
    
    clean_text = preprocess_gutenberg_book(input_path, output_path)
    
    # Show first 500 chars
    print("\nFirst 500 chars of cleaned text:")
    print(clean_text[:500])
