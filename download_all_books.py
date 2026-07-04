"""
Download all popular books from Project Gutenberg.
"""

import sys
import os
import urllib.request
import time

# All popular books to download
BOOKS = [
    {'id': 1342, 'title': 'Pride and Prejudice'},
    {'id': 98, 'title': 'A Tale of Two Cities'},
    {'id': 1661, 'title': 'The Adventures of Sherlock Holmes'},
    {'id': 2701, 'title': 'Moby Dick'},
    {'id': 100, 'title': 'The Complete Works of William Shakespeare'},
    {'id': 1184, 'title': 'The Count of Monte Cristo'},
    {'id': 2600, 'title': 'War and Peace'},
    {'id': 135, 'title': 'Les Miserables'},
    {'id': 996, 'title': 'Don Quixote'},
    {'id': 1727, 'title': 'The Odyssey'},
    {'id': 84, 'title': 'Frankenstein'},
    {'id': 345, 'title': 'Dracula'},
    {'id': 11, 'title': 'Alice in Wonderland'},
    {'id': 74, 'title': 'The Adventures of Tom Sawyer'},
    {'id': 1400, 'title': 'Great Expectations'},
    {'id': 174, 'title': 'The Picture of Dorian Gray'},
    {'id': 1260, 'title': 'Jane Eyre'},
    {'id': 768, 'title': 'Wuthering Heights'},
    {'id': 33, 'title': 'The Scarlet Letter'},
    {'id': 46, 'title': 'A Christmas Carol'},
]

def download_book(book_id, title, output_dir='books'):
    """Download a book from Project Gutenberg."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Gutenberg URL for plain text
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    
    try:
        # Download the book
        response = urllib.request.urlopen(url, timeout=60)
        text = response.read().decode('utf-8', errors='ignore')
        
        # Clean up the title for filename
        safe_title = ''.join(c for c in title if c.isalnum() or c in ' -_').strip()
        safe_title = safe_title[:50]  # Limit length
        
        # Save to file
        filename = f'{output_dir}/{book_id}_{safe_title}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return filename, len(text)
        
    except Exception as e:
        return None, str(e)

def main():
    print('Downloading all popular books from Project Gutenberg...')
    print('=' * 60)
    
    total_size = 0
    downloaded = 0
    failed = 0
    
    for i, book in enumerate(BOOKS, 1):
        book_id = book['id']
        title = book['title']
        
        print(f'[{i}/{len(BOOKS)}] {title}...', end=' ', flush=True)
        
        filename, size = download_book(book_id, title)
        
        if filename:
            print(f'OK ({size:,} chars)')
            total_size += size
            downloaded += 1
        else:
            print(f'FAILED: {size}')
            failed += 1
        
        # Small delay to be polite to Gutenberg
        time.sleep(0.5)
    
    print('=' * 60)
    print(f'Downloaded: {downloaded} books')
    print(f'Failed: {failed} books')
    print(f'Total size: {total_size:,} characters ({total_size/1024/1024:.1f} MB)')
    print()
    print('Books saved to: books/')

if __name__ == '__main__':
    main()
