"""
Download a book from Project Gutenberg.
Usage: python download_book.py <gutenberg_id>

Examples:
    python download_book.py 98  # A Tale of Two Cities
    python download_book.py 1661  # Sherlock Holmes
    python download_book.py 2701  # Moby Dick
"""

import sys
import os
import urllib.request

def download_book(book_id, output_dir='books'):
    """Download a book from Project Gutenberg."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Gutenberg URL for plain text
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    
    print(f'Downloading book {book_id}...')
    print(f'URL: {url}')
    
    try:
        # Download the book
        response = urllib.request.urlopen(url, timeout=60)
        text = response.read().decode('utf-8', errors='ignore')
        
        # Get the title from the first few lines
        lines = text.split('\n')
        title = 'Unknown'
        for line in lines[:20]:
            if 'title' in line.lower() or line.strip():
                title = line.strip()
                if title:
                    break
        
        # Clean up the title for filename
        safe_title = ''.join(c for c in title if c.isalnum() or c in ' -_').strip()
        safe_title = safe_title[:50]  # Limit length
        
        # Save to file
        filename = f'{output_dir}/{book_id}_{safe_title}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f'Downloaded: {filename}')
        print(f'Size: {len(text):,} characters')
        print(f'Lines: {len(lines):,}')
        
        return filename
        
    except Exception as e:
        print(f'Error downloading book {book_id}: {e}')
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python download_book.py <gutenberg_id>')
        print()
        print('Examples:')
        print('  python download_book.py 98  # A Tale of Two Cities')
        print('  python download_book.py 1661  # Sherlock Holmes')
        print('  python download_book.py 2701  # Moby Dick')
        sys.exit(1)
    
    book_id = sys.argv[1]
    download_book(book_id)
