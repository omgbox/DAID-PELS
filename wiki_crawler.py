"""
Crawl Wikipedia by following links inside pages.
Start with seed pages, follow all internal links.
"""

import sys, os, json, time, re
import requests
from urllib.parse import quote, unquote
from collections import deque
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Seed pages to start crawling from
SEEDS = [
    'Python (programming language)', 'Artificial intelligence', 'Machine learning',
    'World War II', 'United States', 'Albert Einstein', 'Evolution', 'DNA',
    'Quantum computing', 'Internet', 'Democracy', 'Mathematics', 'Physics',
    'Chemistry', 'Biology', 'Medicine', 'History', 'Geography', 'Economics',
    'Music', 'Film', 'Art', 'Literature', 'Philosophy', 'Solar system', 'Mars',
    'Black hole', 'Galaxy', 'Bitcoin', 'Blockchain', '5G', 'Virtual reality',
    'Cancer', 'Diabetes', 'Heart disease', 'France', 'Germany', 'Japan',
    'China', 'India', 'Apple Inc.', 'Google', 'Microsoft', 'Amazon',
    'Electricity', 'Engine', 'Motor', 'Computer', 'Software', 'Hardware',
    'Television', 'Radio', 'Telephone', 'Camera', 'Printer', 'Battery',
    'Solar energy', 'Wind power', 'Nuclear power', 'Oil', 'Coal', 'Gas',
    'Steel', 'Concrete', 'Plastic', 'Glass', 'Wood', 'Cotton', 'Silk',
    'Football', 'Basketball', 'Tennis', 'Golf', 'Swimming', 'Running',
    'Rock music', 'Pop music', 'Jazz', 'Classical music', 'Hip hop',
    'Shakespeare', 'Mark Twain', 'Charles Dickens', 'Jane Austen',
    'Leonardo da Vinci', 'Michelangelo', 'Picasso', 'Van Gogh',
    'Newton', 'Einstein', 'Darwin', 'Tesla', 'Edison', 'Ford',
    'NASA', 'SpaceX', 'Tesla Inc', 'Apple', 'Facebook', 'Twitter',
    'Africa', 'Asia', 'Europe', 'North America', 'South America',
    'Ocean', 'Mountain', 'River', 'Desert', 'Forest', 'Island',
    'Volcano', 'Earthquake', 'Tsunami', 'Hurricane', 'Tornado',
    'DNA', 'RNA', 'Protein', 'Cell', 'Tissue', 'Organ',
    'Atom', 'Molecule', 'Electron', 'Proton', 'Neutron',
    'Energy', 'Force', 'Motion', 'Gravity', 'Magnetism',
    'Light', 'Sound', 'Heat', 'Temperature', 'Pressure',
]


class WikiCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'BookBot-Crawler/1.0'})
        self.visited = set()
        self.queue = deque()
        self.pages = []
    
    def get_links(self, title):
        """Get internal links from a Wikipedia page."""
        try:
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'links',
                'pllimit': 100,
                'plnamespace': 0,  # Main namespace only
                'format': 'json'
            }
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get('query', {}).get('pages', {})
                for page_id, page_data in pages.items():
                    links = page_data.get('links', [])
                    return [l['title'] for l in links]
        except Exception:
            pass
        return []
    
    def get_summary(self, title):
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
    
    def crawl(self, max_pages=500):
        """Crawl Wikipedia starting from seed pages."""
        # Add seeds to queue
        for seed in SEEDS:
            if seed not in self.visited:
                self.queue.append(seed)
        
        print(f"Starting crawl with {len(SEEDS)} seed pages...")
        print(f"Target: {max_pages} pages\n")
        
        while self.queue and len(self.visited) < max_pages:
            # Get next page from queue
            title = self.queue.popleft()
            
            if title in self.visited:
                continue
            
            # Get page summary
            page = self.get_summary(title)
            
            if not page or not page['extract']:
                continue
            
            # Mark as visited
            self.visited.add(title)
            self.pages.append(page)
            
            # Get links from page
            links = self.get_links(title)
            
            # Add unvisited links to queue
            for link in links:
                if link not in self.visited and link not in self.queue:
                    # Filter out non-article pages
                    if not link.startswith('Wikipedia:') and not link.startswith('Category:'):
                        self.queue.append(link)
            
            # Progress
            if len(self.visited) % 25 == 0:
                print(f"  Crawled: {len(self.visited)} pages | Queue: {len(self.queue)} | Current: {title[:40]}")
            
            # Rate limit
            time.sleep(0.05)
        
        return self.pages


def train():
    print("=" * 60)
    print("  Wikipedia Link Crawler + Training")
    print("=" * 60)
    
    # Crawl Wikipedia
    crawler = WikiCrawler()
    pages = crawler.crawl(max_pages=500)
    
    print(f"\nCrawled {len(pages)} pages. Training neural networks...\n")
    
    # Train on crawled pages (load existing models first)
    ai = ConversationalAI()
    
    # Force-create all neural networks
    base = os.path.dirname(os.path.abspath(__file__))
    ai._get_topic_extractor()
    ai._get_neural_mapper()
    ai._get_intent_classifier()
    ai._get_response_selector()
    
    # Load existing models
    if ai._topic_extractor:
        ai._topic_extractor.load(os.path.join(base, 'topic_scores.json'))
        print(f"  Loaded topic_extractor: {len(ai._topic_extractor.word_scores)} words")
    if ai._neural_mapper:
        ai._neural_mapper.load(os.path.join(base, 'wiki_mappings.json'))
        print(f"  Loaded neural_mapper: {len(ai._neural_mapper.learned_mappings)} mappings")
    if ai._intent_classifier:
        ai._intent_classifier.load(os.path.join(base, 'intent_scores.json'))
        print(f"  Loaded intent_classifier: {len(ai._intent_classifier.intent_scores)} patterns")
    
    total_trained = 0
    
    # Query templates
    templates = [
        '{title}',
        'what is {title}?',
        'tell me about {title}',
        'explain {title}',
        'how does {title} work?',
        'who invented {title}?',
        'when was {title} created?',
        'why is {title} important?',
    ]
    
    for i, page in enumerate(pages):
        title = page['title']
        
        # Train on each template
        for template in templates:
            query = template.format(title=title)
            
            # Train topic extractor
            if ai._topic_extractor:
                ai._topic_extractor.train(query, title, positive=True)
            
            # Train Wikipedia mapper
            if ai._neural_mapper:
                ai._neural_mapper.train(query, title, positive=True)
                # Debug: print first few
                if len(ai._neural_mapper.learned_mappings) <= 10:
                    print(f"    Added mapping: {query[:30]} -> {title[:30]}")
            
            # Train intent classifier
            if ai._intent_classifier:
                ai._intent_classifier.train(query, 'question', positive=True)
            
            total_trained += 1
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Trained: {i+1}/{len(pages)} pages ({total_trained} queries)")
    
    # Debug: check what was trained
    print(f"\n  ai._neural_mapper is None: {ai._neural_mapper is None}")
    if ai._neural_mapper:
        print(f"  Neural mapper learned_mappings: {len(ai._neural_mapper.learned_mappings)} mappings")
        print(f"  Sample mappings: {list(ai._neural_mapper.learned_mappings.items())[:5]}")
    
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
    print(f"  Complete!")
    print(f"  Pages crawled: {len(pages)}")
    print(f"  Total queries trained: {total_trained}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
