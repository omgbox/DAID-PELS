"""
List popular public domain books from Project Gutenberg.
"""

# List of popular public domain books from Project Gutenberg
books = [
    {'title': 'Pride and Prejudice', 'author': 'Jane Austen', 'id': 1342, 'size': '126KB'},
    {'title': 'A Tale of Two Cities', 'author': 'Charles Dickens', 'id': 98, 'size': '200KB'},
    {'title': 'The Adventures of Sherlock Holmes', 'author': 'Arthur Conan Doyle', 'id': 1661, 'size': '180KB'},
    {'title': 'Moby Dick; or The Whale', 'author': 'Herman Melville', 'id': 2701, 'size': '250KB'},
    {'title': 'The Complete Works of William Shakespeare', 'author': 'William Shakespeare', 'id': 100, 'size': '300KB'},
    {'title': 'The Count of Monte Cristo', 'author': 'Alexandre Dumas', 'id': 1184, 'size': '350KB'},
    {'title': 'War and Peace', 'author': 'Leo Tolstoy', 'id': 2600, 'size': '400KB'},
    {'title': 'Les Miserables', 'author': 'Victor Hugo', 'id': 135, 'size': '380KB'},
    {'title': 'Don Quixote', 'author': 'Miguel de Cervantes', 'id': 996, 'size': '280KB'},
    {'title': 'The Odyssey', 'author': 'Homer', 'id': 1727, 'size': '150KB'},
    {'title': 'Frankenstein', 'author': 'Mary Shelley', 'id': 84, 'size': '100KB'},
    {'title': 'Dracula', 'author': 'Bram Stoker', 'id': 345, 'size': '180KB'},
    {'title': 'Alice in Wonderland', 'author': 'Lewis Carroll', 'id': 11, 'size': '80KB'},
    {'title': 'The Adventures of Tom Sawyer', 'author': 'Mark Twain', 'id': 74, 'size': '130KB'},
    {'title': 'Great Expectations', 'author': 'Charles Dickens', 'id': 1400, 'size': '200KB'},
    {'title': 'The Picture of Dorian Gray', 'author': 'Oscar Wilde', 'id': 174, 'size': '120KB'},
    {'title': 'Jane Eyre', 'author': 'Charlotte Bronte', 'id': 1260, 'size': '180KB'},
    {'title': 'Wuthering Heights', 'author': 'Emily Bronte', 'id': 768, 'size': '150KB'},
    {'title': 'The Scarlet Letter', 'author': 'Nathaniel Hawthorne', 'id': 33, 'size': '120KB'},
    {'title': 'A Christmas Carol', 'author': 'Charles Dickens', 'id': 46, 'size': '60KB'},
]

print('Popular Public Domain Books from Project Gutenberg')
print('=' * 70)
print(f'{"Title":<45} {"Author":<25} {"Size":<8}')
print('-' * 70)
for book in books:
    print(f'{book["title"]:<45} {book["author"]:<25} {book["size"]:<8}')

print()
print('To download any book, run:')
print('  python download_book.py <gutenberg_id>')
print()
print('Example:')
print('  python download_book.py 98  # A Tale of Two Cities')
print('  python download_book.py 1661  # Sherlock Holmes')
