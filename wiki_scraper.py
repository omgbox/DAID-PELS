"""
Scrape Wikipedia systematically for training data.
Uses category system to get thousands of real articles.
"""

import sys, os, json, time, random
import requests
from urllib.parse import quote
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Wikipedia categories to scrape
CATEGORIES = [
    'Programming languages', 'Computer science', 'Mathematics',
    'Physics', 'Chemistry', 'Biology', 'Medicine',
    'History', 'Geography', 'Economics', 'Psychology',
    'Philosophy', 'Art', 'Music', 'Literature',
    'Technology', 'Engineering', 'Agriculture',
    'Countries', 'Cities', 'People', 'Organizations',
    'Diseases', 'Animals', 'Plants', 'Minerals',
    'Space', 'Astronomy', 'Geology', 'Ecology',
    'Sports', 'Games', 'Food', 'Fashion',
    'Religion', 'Mythology', 'Folklore',
    'Education', 'Law', 'Politics', 'Military',
    'Architecture', 'Transport', 'Energy',
    'Internet', 'Software', 'Hardware',
]

# Query templates
QUERY_TEMPLATES = [
    '{title}',
    'what is {title}?',
    'tell me about {title}',
    'explain {title}',
    'how does {title} work?',
    'who invented {title}?',
    'when was {title} created?',
    'why is {title} important?',
    'what are {title} used for?',
    'describe {title}',
]


class WikiScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'BookBot-Scraper/1.0'})
        self.base_url = 'https://en.wikipedia.org/w/api.php'
    
    def get_category_members(self, category, limit=50):
        """Get articles from a Wikipedia category."""
        try:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': limit,
                'cmtype': 'page',
                'format': 'json'
            }
            resp = self.session.get(self.base_url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                members = data.get('query', {}).get('categorymembers', [])
                return [m['title'] for m in members]
        except Exception as e:
            pass
        return []
    
    def get_page_summary(self, title):
        """Get Wikipedia page summary."""
        try:
            encoded = quote(title.replace(' ', '_'))
            url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}'
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'title': data.get('title', title),
                    'extract': data.get('extract', ''),
                    'description': data.get('description', ''),
                }
        except Exception:
            pass
        return None
    
    def get_subcategories(self, category, limit=10):
        """Get subcategories from a category."""
        try:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': limit,
                'cmtype': 'subcat',
                'format': 'json'
            }
            resp = self.session.get(self.base_url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                members = data.get('query', {}).get('categorymembers', [])
                return [m['title'].replace('Category:', '') for m in members]
        except Exception:
            pass
        return []


def scrape():
    print("=" * 60)
    print("  Wikipedia Systematic Scraping")
    print("=" * 60)
    
    ai = ConversationalAI()
    scraper = WikiScraper()
    
    all_pages = set()
    total_trained = 0
    
    print(f"\nScraping {len(CATEGORIES)} categories...")
    print("This will take a while - downloading real Wikipedia data.\n")
    
    for cat_idx, category in enumerate(CATEGORIES):
        print(f"\n[{cat_idx+1}/{len(CATEGORIES)}] Category: {category}")
        
        # Get articles from category
        pages = scraper.get_category_members(category, limit=100)
        
        if not pages:
            print(f"  No pages found, trying subcategories...")
            subcats = scraper.get_subcategories(category, limit=5)
            for subcat in subcats:
                sub_pages = scraper.get_category_members(subcat, limit=50)
                pages.extend(sub_pages)
                time.sleep(0.1)
        
        print(f"  Found {len(pages)} pages")
        
        # Download and train on each page
        for page_idx, page_title in enumerate(pages[:50]):  # Limit to 50 per category
            if page_title in all_pages:
                continue
            
            # Get page summary
            page = scraper.get_page_summary(page_title)
            
            if not page or not page['extract']:
                continue
            
            all_pages.add(page_title)
            title = page['title']
            
            # Train on each query template
            for template in QUERY_TEMPLATES:
                query = template.format(title=title)
                
                # Train topic extractor
                if ai._topic_extractor:
                    ai._topic_extractor.train(query, title, positive=True)
                
                # Train Wikipedia mapper
                if ai._neural_mapper:
                    ai._neural_mapper.train(query, title, positive=True)
                
                # Train intent classifier
                if ai._intent_classifier:
                    ai._intent_classifier.train(query, 'question', positive=True)
                
                total_trained += 1
            
            # Progress
            if (page_idx + 1) % 10 == 0:
                print(f"    [{page_idx+1}/{len(pages)}] {title}")
            
            # Rate limit
            time.sleep(0.05)
    
    # Save
    print(f"\n{'='*60}")
    print("Saving trained models...")
    base = os.path.dirname(os.path.abspath(__file__))
    
    if ai._topic_extractor:
        path = os.path.join(base, 'topic_scores.json')
        ai._topic_extractor.save(path)
        with open(path) as f:
            data = json.load(f)
        print(f"  topic_scores.json: {len(data.get('word_scores', {}))} words")
    
    if ai._neural_mapper:
        path = os.path.join(base, 'wiki_mappings.json')
        ai._neural_mapper.save(path)
        with open(path) as f:
            data = json.load(f)
        print(f"  wiki_mappings.json: {len(data.get('learned_mappings', {}))} mappings")
    
    if ai._intent_classifier:
        path = os.path.join(base, 'intent_scores.json')
        ai._intent_classifier.save(path)
        with open(path) as f:
            data = json.load(f)
        print(f"  intent_scores.json: {len(data.get('intent_scores', {}))} patterns")
    
    print(f"\n{'='*60}")
    print(f"  Scraping complete!")
    print(f"  Pages scraped: {len(all_pages)}")
    print(f"  Total queries trained: {total_trained}")
    print(f"{'='*60}")


if __name__ == '__main__':
    scrape()
