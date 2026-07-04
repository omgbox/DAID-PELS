"""
Pre-train neural networks with synthetic data.
Generates common query patterns and trains all networks.
"""

import sys, os, json, random
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Common topics and their Wikipedia pages
TOPICS = {
    # Programming
    'python': 'Python (programming language)',
    'javascript': 'JavaScript',
    'rust': 'Rust (programming language)',
    'java': 'Java (programming language)',
    'c++': 'C++',
    'golang': 'Go (programming language)',
    'typescript': 'TypeScript',
    'ruby': 'Ruby (programming language)',
    'php': 'PHP',
    'swift': 'Swift (programming language)',
    'kotlin': 'Kotlin',
    'scala': 'Scala (programming language)',
    'haskell': 'Haskell',
    'elixir': 'Elixir (programming language)',
    'erlang': 'Erlang',
    
    # Science
    'quantum computing': 'Quantum computing',
    'machine learning': 'Machine learning',
    'artificial intelligence': 'Artificial intelligence',
    'deep learning': 'Deep learning',
    'neural network': 'Artificial neural network',
    'blockchain': 'Blockchain',
    'cryptocurrency': 'Cryptocurrency',
    'bitcoin': 'Bitcoin',
    'ethereum': 'Ethereum',
    
    # History
    'world war 2': 'World War II',
    'world war 1': 'World War I',
    'roman empire': 'Roman Empire',
    'ancient egypt': 'Ancient Egypt',
    'middle ages': 'Middle Ages',
    'renaissance': 'Renaissance',
    'industrial revolution': 'Industrial Revolution',
    'cold war': 'Cold War',
    
    # Geography
    'united states': 'United States',
    'united kingdom': 'United Kingdom',
    'european union': 'European Union',
    'china': 'China',
    'japan': 'Japan',
    'india': 'India',
    'brazil': 'Brazil',
    'australia': 'Australia',
    
    # Science
    'evolution': 'Evolution',
    'dna': 'DNA',
    'climate change': 'Climate change',
    'global warming': 'Global warming',
    'renewable energy': 'Renewable energy',
    'solar energy': 'Solar energy',
    'nuclear energy': 'Nuclear power',
    
    # Technology
    'internet': 'Internet',
    'world wide web': 'World Wide Web',
    'artificial intelligence': 'Artificial intelligence',
    'machine learning': 'Machine learning',
    'virtual reality': 'Virtual reality',
    'augmented reality': 'Augmented reality',
    '5g': '5G',
    'wifi': 'Wi-Fi',
    
    # Health
    'vaccine': 'Vaccine',
    'antibiotic': 'Antibiotic',
    'cancer': 'Cancer',
    'diabetes': 'Diabetes',
    'heart disease': 'Heart disease',
    'mental health': 'Mental health',
    'depression': 'Depression',
    'anxiety': 'Anxiety',
    
    # Culture
    'music': 'Music',
    'film': 'Film',
    'art': 'Art',
    'literature': 'Literature',
    'philosophy': 'Philosophy',
    'religion': 'Religion',
    'language': 'Language',
    'education': 'Education',
}

# Query templates
TEMPLATES = [
    'what is {topic}?',
    'who invented {topic}?',
    'when was {topic} invented?',
    'how does {topic} work?',
    'why is {topic} important?',
    'tell me about {topic}',
    'explain {topic}',
    'what are the benefits of {topic}?',
    'what are the risks of {topic}?',
    'how to learn {topic}?',
    'what is the history of {topic}?',
    'who discovered {topic}?',
    'what is {topic} used for?',
    'what is the future of {topic}?',
    'compare {topic} with alternatives',
]

# Intent templates
INTENT_TEMPLATES = {
    'greeting': ['hello', 'hi', 'hey', 'howdy', 'good morning', 'good evening'],
    'farewell': ['bye', 'goodbye', 'see you', 'take care', 'good night'],
    'personal': ['i like {topic}', 'i love {topic}', 'i hate {topic}', 'i am learning {topic}'],
    'emotional': ['i feel happy', 'i feel sad', 'i am excited', 'i am tired'],
}


def pretrain():
    print("=" * 60)
    print("  Pre-training Neural Networks")
    print("=" * 60)
    
    ai = ConversationalAI()
    
    # Generate queries
    queries = []
    
    # Topic-based queries
    for topic, wiki_page in TOPICS.items():
        for template in random.sample(TEMPLATES, min(5, len(TEMPLATES))):
            queries.append({
                'query': template.format(topic=topic),
                'topic': topic,
                'wiki_page': wiki_page,
                'intent': 'question'
            })
    
    # Intent-based queries
    for intent, templates in INTENT_TEMPLATES.items():
        for template in templates:
            queries.append({
                'query': template,
                'topic': '',
                'wiki_page': '',
                'intent': intent
            })
    
    # Shuffle
    random.shuffle(queries)
    
    print(f"\nGenerated {len(queries)} training queries")
    print(f"Topics: {len(TOPICS)}")
    print(f"Query templates: {len(TEMPLATES)}")
    
    # Train
    print("\nTraining...")
    for i, q in enumerate(queries):
        query = q['query']
        topic = q['topic']
        wiki_page = q['wiki_page']
        intent = q['intent']
        
        # Train topic extractor
        if ai._topic_extractor and topic:
            ai._topic_extractor.train(query, topic, positive=True)
        
        # Train Wikipedia mapper
        if ai._neural_mapper and wiki_page:
            ai._neural_mapper.train(query, wiki_page, positive=True)
        
        # Train intent classifier
        if ai._intent_classifier:
            ai._intent_classifier.train(query, intent, positive=True)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Trained {i + 1}/{len(queries)} queries...")
    
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
    
    if ai._response_selector:
        ai._response_selector.save(os.path.join(base, 'response_scores.json'))
        print(f"  response_scores.json: {len(ai._response_selector.response_scores)} preferences")
    
    print("\n" + "=" * 60)
    print("  Pre-training complete!")
    print("=" * 60)


if __name__ == '__main__':
    pretrain()
