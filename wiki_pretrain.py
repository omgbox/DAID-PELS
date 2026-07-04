"""
Pre-train neural networks with REAL Wikipedia data.
Downloads summaries for common topics and trains all networks.
"""

import sys, os, json, time, random
import requests
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Wikipedia topics to train on
TOPICS = [
    # Programming
    'Python (programming language)', 'JavaScript', 'Rust (programming language)',
    'Java (programming language)', 'C++', 'Go (programming language)',
    'TypeScript', 'Ruby (programming language)', 'PHP', 'Swift (programming language)',
    'Kotlin', 'Scala (programming language)', 'Haskell', 'Elixir (programming language)',
    'Machine learning', 'Deep learning', 'Artificial neural network',
    
    # Science
    'Quantum computing', 'Artificial intelligence', 'Blockchain', 'Bitcoin',
    'Ethereum', 'Machine learning', 'Natural language processing',
    'Computer vision', 'Robotics', 'Internet of Things',
    
    # History
    'World War II', 'World War I', 'Roman Empire', 'Ancient Egypt',
    'Middle Ages', 'Renaissance', 'Industrial Revolution', 'Cold War',
    'French Revolution', 'American Revolution',
    
    # Geography
    'United States', 'United Kingdom', 'European Union', 'China',
    'Japan', 'India', 'Brazil', 'Australia', 'Canada', 'Germany',
    
    # Science
    'Evolution', 'DNA', 'Climate change', 'Global warming',
    'Renewable energy', 'Solar energy', 'Nuclear power', 'Electricity',
    'Photosynthesis', 'Cell biology',
    
    # Technology
    'Internet', 'World Wide Web', '5G', 'Wi-Fi', 'Bluetooth',
    'Cloud computing', 'Virtual reality', 'Augmented reality',
    'Autonomous vehicle', 'Space exploration',
    
    # Health
    'Vaccine', 'Antibiotic', 'Cancer', 'Diabetes', 'Heart disease',
    'Mental health', 'Depression', 'Anxiety', 'Exercise', 'Nutrition',
    
    # Culture
    'Music', 'Film', 'Art', 'Literature', 'Philosophy', 'Religion',
    'Language', 'Education', 'Economics', 'Psychology',
    
    # Nature
    'Ocean', 'Forest', 'Mountain', 'River', 'Desert', 'Arctic',
    'Biodiversity', 'Ecology', 'Climate', 'Weather',
    
    # Space
    'Solar system', 'Mars', 'Moon', 'Sun', 'Galaxy', 'Black hole',
    'Big Bang', 'Dark matter', 'Dark energy', 'Exoplanet',
    
    # Math
    'Mathematics', 'Algebra', 'Calculus', 'Statistics', 'Geometry',
    'Prime number', 'Infinity', 'Fractal', 'Graph theory',
]


def get_wikipedia_summary(topic):
    """Get Wikipedia summary for a topic."""
    try:
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{topic}'
        headers = {'User-Agent': 'BookBot-Pretrain/1.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('extract', '')
    except Exception as e:
        pass
    return None


def get_wikipedia_search(query):
    """Search Wikipedia and return top results."""
    try:
        url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': 5
        }
        headers = {'User-Agent': 'BookBot-Pretrain/1.0'}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [s['title'] for s in data.get('query', {}).get('search', [])]
    except Exception:
        pass
    return []


def pretrain():
    print("=" * 60)
    print("  Wikipedia Pre-training")
    print("=" * 60)
    
    ai = ConversationalAI()
    
    # Query templates
    templates = [
        '{topic}',
        'what is {topic}?',
        'who invented {topic}?',
        'how does {topic} work?',
        'tell me about {topic}',
        'explain {topic}',
    ]
    
    total = len(TOPICS) * len(templates)
    trained = 0
    errors = 0
    
    print(f"\nTopics: {len(TOPICS)}")
    print(f"Templates per topic: {len(templates)}")
    print(f"Total queries: {total}")
    print("\nDownloading from Wikipedia and training...\n")
    
    for topic in TOPICS:
        # Get Wikipedia page
        summary = get_wikipedia_summary(topic)
        
        if not summary:
            print(f"  [SKIP] {topic} - not found")
            errors += 1
            continue
        
        print(f"  [OK] {topic} ({len(summary)} chars)")
        
        # Train on each template
        for template in templates:
            query = template.format(topic=topic.replace(' (programming language)', '').replace(' (programming language)', ''))
            
            # Train topic extractor
            if ai._topic_extractor:
                ai._topic_extractor.train(query, topic, positive=True)
            
            # Train Wikipedia mapper
            if ai._neural_mapper:
                ai._neural_mapper.train(query, topic, positive=True)
            
            # Train intent classifier
            if ai._intent_classifier:
                ai._intent_classifier.train(query, 'question', positive=True)
            
            trained += 1
        
        # Rate limit
        time.sleep(0.1)
        
        # Progress
        progress = (trained + errors) / total * 100
        print(f"  Progress: {progress:.1f}% ({trained}/{total})")
    
    # Also train search results
    print("\nTraining on search results...")
    search_queries = [
        'programming', 'science', 'history', 'geography', 'technology',
        'health', 'culture', 'nature', 'space', 'math',
        'python programming', 'machine learning', 'climate change',
        'world war', 'solar system', 'artificial intelligence',
    ]
    
    for query in search_queries:
        results = get_wikipedia_search(query)
        if results:
            for title in results[:3]:
                if ai._neural_mapper:
                    ai._neural_mapper.train(query, title, positive=True)
                trained += 1
            print(f"  Search '{query}': {len(results)} results")
        time.sleep(0.1)
    
    # Save
    print("\nSaving models...")
    base = os.path.dirname(os.path.abspath(__file__))
    
    if ai._topic_extractor:
        ai._topic_extractor.save(os.path.join(base, 'topic_scores.json'))
        print(f"  topic_scores.json: {len(ai._topic_extractor.word_scores)} words")
    
    if ai._neural_mapper:
        ai._neural_mapper.save(os.path.join(base, 'wiki_mappings.json'))
        print(f"  wiki_mappings.json: {len(ai._neural_mapper.learned_mappings)} mappings")
    
    if ai._intent_classifier:
        ai._intent_classifier.save(os.path.join(base, 'intent_scores.json'))
        print(f"  intent_scores.json: {len(ai._intent_classifier.intent_scores)} patterns")
    
    print("\n" + "=" * 60)
    print(f"  Pre-training complete!")
    print(f"  Trained: {trained} queries")
    print(f"  Errors: {errors}")
    print("=" * 60)


if __name__ == '__main__':
    pretrain()
