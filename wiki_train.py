"""
Train on REAL Wikipedia data - factual records.
Downloads actual Wikipedia pages and trains all networks.
"""

import sys, os, json, time, random
import requests
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Real Wikipedia pages to download
WIKI_PAGES = [
    # Programming - real pages
    'Python (programming language)', 'JavaScript', 'Rust (programming language)',
    'Java (programming language)', 'C++', 'Go (programming language)',
    'TypeScript', 'Ruby (programming language)', 'PHP', 'Swift (programming language)',
    'Kotlin', 'Scala (programming language)', 'Haskell', 'Elixir (programming language)',
    'Machine learning', 'Deep learning', 'Artificial neural network',
    'Computer science', 'Software engineering', 'Algorithm',
    
    # Science - real pages
    'Quantum computing', 'Artificial intelligence', 'Blockchain', 'Bitcoin',
    'Ethereum', 'Natural language processing', 'Computer vision', 'Robotics',
    'Internet of Things', 'Cloud computing', 'Data science', 'Big data',
    'Cybersecurity', 'Cryptography', 'Networking', 'Database',
    
    # History - real pages
    'World War II', 'World War I', 'Roman Empire', 'Ancient Egypt',
    'Middle Ages', 'Renaissance', 'Industrial Revolution', 'Cold War',
    'French Revolution', 'American Revolution', 'Ancient Greece',
    'Medieval Europe', 'Age of Enlightenment', 'Colonialism',
    
    # Geography - real pages
    'United States', 'United Kingdom', 'European Union', 'China',
    'Japan', 'India', 'Brazil', 'Australia', 'Canada', 'Germany',
    'France', 'Italy', 'Spain', 'Mexico', 'South Korea', 'Russia',
    
    # Science - real pages
    'Evolution', 'DNA', 'Climate change', 'Global warming',
    'Renewable energy', 'Solar energy', 'Nuclear power', 'Electricity',
    'Photosynthesis', 'Cell biology', 'Genetics', 'Ecology',
    'Chemistry', 'Physics', 'Biology', 'Mathematics',
    
    # Technology - real pages
    'Internet', 'World Wide Web', '5G', 'Wi-Fi', 'Bluetooth',
    'Virtual reality', 'Augmented reality', 'Autonomous vehicle',
    'Space exploration', 'Mars', 'International Space Station',
    
    # Health - real pages
    'Vaccine', 'Antibiotic', 'Cancer', 'Diabetes', 'Heart disease',
    'Mental health', 'Depression', 'Anxiety', 'Exercise', 'Nutrition',
    'Immunology', 'Epidemiology', 'Neuroscience',
    
    # Culture - real pages
    'Music', 'Film', 'Art', 'Literature', 'Philosophy', 'Religion',
    'Language', 'Education', 'Economics', 'Psychology', 'Sociology',
    
    # Nature - real pages
    'Ocean', 'Forest', 'Mountain', 'River', 'Desert', 'Arctic',
    'Biodiversity', 'Ecology', 'Climate', 'Weather', 'Volcano', 'Earthquake',
    
    # Space - real pages
    'Solar system', 'Mars', 'Moon', 'Sun', 'Galaxy', 'Black hole',
    'Big Bang', 'Dark matter', 'Dark energy', 'Exoplanet', 'Nebula',
    
    # Math - real pages
    'Algebra', 'Calculus', 'Statistics', 'Geometry',
    'Prime number', 'Infinity', 'Fractal', 'Graph theory',
    'Topology', 'Number theory',
    
    # People - real pages
    'Albert Einstein', 'Isaac Newton', 'Nikola Tesla', 'Leonardo da Vinci',
    'Charles Darwin', 'Marie Curie', 'Alan Turing', 'Steve Jobs',
    'Bill Gates', 'Elon Musk', 'Mark Zuckerberg', 'Jeff Bezos',
    
    # Companies - real pages
    'Apple Inc.', 'Google', 'Microsoft', 'Amazon', 'Tesla Inc.',
    'Meta Platforms', 'Netflix', 'SpaceX', 'OpenAI', 'NVIDIA',
    
    # Products - real pages
    'iPhone', 'Android (operating system)', 'Windows', 'macOS', 'Linux',
    'Twitter', 'Facebook', 'Instagram', 'YouTube', 'TikTok',
    
    # Concepts - real pages
    'Democracy', 'Capitalism', 'Socialism', 'Communism', 'Liberalism',
    'Human rights', 'Climate change', 'Sustainability', 'Globalization',
    'Inequality', 'Poverty', 'Education', 'Healthcare',
]


def download_page(title):
    """Download Wikipedia page content."""
    try:
        # URL encode the title
        encoded = title.replace(' ', '_')
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}'
        headers = {'User-Agent': 'BookBot-WikiTrain/1.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'title': data.get('title', title),
                'extract': data.get('extract', ''),
                'description': data.get('description', ''),
            }
    except Exception as e:
        pass
    return None


def train():
    print("=" * 60)
    print("  Wikipedia Real Data Training")
    print("=" * 60)
    
    ai = ConversationalAI()
    
    total = len(WIKI_PAGES)
    success = 0
    failed = 0
    total_trained = 0
    
    print(f"\nDownloading {total} Wikipedia pages...")
    print("Training neural networks on real data...\n")
    
    for i, page_title in enumerate(WIKI_PAGES):
        # Download page
        page = download_page(page_title)
        
        if not page or not page['extract']:
            print(f"  [{i+1}/{total}] SKIP: {page_title}")
            failed += 1
            continue
        
        title = page['title']
        extract = page['extract']
        
        # Generate training queries from the page
        queries = [
            title,
            f'what is {title.lower()}?',
            f'tell me about {title.lower()}',
            f'explain {title.lower()}',
            f'how does {title.lower()} work',
            f'why is {title.lower()} important',
            f'what is the history of {title.lower()}',
            f'who invented {title.lower()}',
            f'when was {title.lower()} created',
            f'what are {title.lower()} used for',
        ]
        
        # Train on each query
        for query in queries:
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
        
        success += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] OK: {title} ({len(extract)} chars)")
        
        # Rate limit
        time.sleep(0.1)
    
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
    print(f"  Training complete!")
    print(f"  Pages downloaded: {success}")
    print(f"  Pages failed: {failed}")
    print(f"  Total queries trained: {total_trained}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
