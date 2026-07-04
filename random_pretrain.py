"""
Random pre-training with diverse topics and queries.
Generates thousands of training examples from Wikipedia.
"""

import sys, os, json, time, random
import requests
sys.path.insert(0, r'C:\projects')

from bookbot.query.conversational_ai import ConversationalAI

# Random topic categories
CATEGORIES = {
    'animals': ['Dog', 'Cat', 'Elephant', 'Lion', 'Tiger', 'Bear', 'Wolf', 'Eagle', 'Shark', 'Dolphin',
                'Whale', 'Penguin', 'Giraffe', 'Zebra', 'Monkey', 'Snake', 'Frog', 'Butterfly', 'Bee', 'Ant'],
    'countries': ['France', 'Germany', 'Italy', 'Spain', 'Mexico', 'Brazil', 'Argentina', 'Egypt', 'Turkey', 'Thailand',
                  'Vietnam', 'Indonesia', 'Philippines', 'Nigeria', 'Kenya', 'South Africa', 'Morocco', 'Peru', 'Chile', 'Colombia'],
    'cities': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Madrid', 'Rome', 'Sydney', 'Toronto', 'Dubai',
               'Singapore', 'Seoul', 'Bangkok', 'Amsterdam', 'Vienna', 'Prague', 'Barcelona', 'Lisbon', 'Stockholm', 'Oslo'],
    'foods': ['Pizza', 'Sushi', 'Tacos', 'Pasta', 'Curry', 'Burger', 'Steak', 'Salad', 'Soup', 'Bread',
              'Cheese', 'Chocolate', 'Ice cream', 'Coffee', 'Tea', 'Wine', 'Beer', 'Juice', 'Water', 'Milk'],
    'sports': ['Football', 'Basketball', 'Tennis', 'Golf', 'Swimming', 'Running', 'Cycling', 'Boxing', 'MMA', 'Wrestling',
               'Cricket', 'Baseball', 'Hockey', 'Volleyball', 'Rugby', 'Skiing', 'Surfing', 'Skateboarding', 'Yoga', 'Gymnastics'],
    'music': ['Rock', 'Pop', 'Jazz', 'Classical', 'Hip hop', 'Electronic', 'Country', 'Blues', 'Reggae', 'Metal',
              'Folk', 'R&B', 'Punk', 'Indie', 'Alternative', 'Opera', 'Gospel', 'Latin', 'Funk', 'Soul'],
    'movies': ['Titanic', 'Avatar', 'Star Wars', 'Jurassic Park', 'The Matrix', 'Inception', 'Interstellar', 'Joker', 'Frozen', 'Toy Story',
               'The Lion King', 'Finding Nemo', 'Shrek', 'Harry Potter', 'Lord of the Rings', 'The Godfather', 'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'The Dark Knight'],
    'books': ['Harry Potter', 'Lord of the Rings', 'The Hobbit', '1984', 'To Kill a Mockingbird', 'Pride and Prejudice', 
              'The Great Gatsby', 'Animal Farm', 'Brave New World', 'The Catcher in the Rye',
              'Don Quixote', 'Moby Dick', 'War and Peace', 'Crime and Punishment', 'The Odyssey'],
    'inventions': ['Internet', 'Telephone', 'Television', 'Computer', 'Airplane', 'Car', 'Bicycle', 'Camera', 'Printer', 'Light bulb',
                   'Electricity', 'Radio', 'Nuclear power', 'Solar panel', 'Battery', 'Transistor', 'Microchip', 'GPS', 'Touchscreen', 'QR code'],
    'diseases': ['Cancer', 'Diabetes', 'Heart disease', 'Stroke', 'Alzheimer', 'Parkinson', 'Asthma', 'Arthritis', 'Obesity', 'Hypertension',
                 'Influenza', 'Malaria', 'HIV', 'Tuberculosis', 'Pneumonia', 'Bronchitis', 'Epilepsy', 'Migraine', 'Allergy', 'Anemia'],
    'planets': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Moon',
                'Sun', 'Mars', 'Jupiter', 'Saturn', 'Asteroid', 'Comet', 'Meteor', 'Galaxy', 'Nebula', 'Black hole'],
    'emotions': ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Trust', 'Anticipation', 'Love', 'Jealousy',
                 'Pride', 'Shame', 'Guilt', 'Hope', 'Despair', 'Joy', 'Anxiety', 'Excitement', 'Boredom', 'Curiosity'],
    'professions': ['Doctor', 'Teacher', 'Engineer', 'Lawyer', 'Artist', 'Musician', 'Writer', 'Scientist', 'Programmer', 'Nurse',
                    'Pilot', 'Chef', 'Architect', 'Journalist', 'Farmer', 'Mechanic', 'Dentist', 'Pharmacist', 'Accountant', 'Designer'],
    'materials': ['Wood', 'Metal', 'Plastic', 'Glass', 'Cotton', 'Silk', 'Leather', 'Rubber', 'Concrete', 'Steel',
                  'Aluminum', 'Copper', 'Gold', 'Silver', 'Diamond', 'Carbon', 'Silicon', 'Titanium', 'Uranium', 'Lithium'],
    'emotions_en': ['Happy', 'Sad', 'Angry', 'Scared', 'Surprised', 'Disgusted', 'Excited', 'Tired', 'Bored', 'Curious',
                    'Proud', 'Embarrassed', 'Confused', 'Grateful', 'Jealous', 'Lonely', 'Peaceful', 'Stressed', 'Relaxed', 'Motivated'],
}

# Query patterns
PATTERNS = [
    '{topic}',
    'what is {topic}?',
    'tell me about {topic}',
    'how does {topic} work?',
    'why is {topic} important?',
    'who invented {topic}?',
    'when was {topic} created?',
    'where is {topic} from?',
    'what are the benefits of {topic}?',
    'what are the types of {topic}?',
    'explain {topic}',
    'describe {topic}',
    'give me information about {topic}',
    'what do you know about {topic}?',
    'can you tell me about {topic}?',
]

# Intent patterns
INTENTS = {
    'greeting': ['hello', 'hi', 'hey', 'howdy', 'good morning', 'good evening', 'whats up', 'yo'],
    'farewell': ['bye', 'goodbye', 'see you', 'take care', 'good night', 'later', 'cya'],
    'personal': ['i like', 'i love', 'i hate', 'i enjoy', 'i prefer', 'i think', 'i believe', 'i feel'],
    'emotional': ['i am happy', 'i am sad', 'i am angry', 'i am excited', 'i am tired', 'i am bored'],
}


def get_wikipedia_summary(topic):
    """Get Wikipedia summary."""
    try:
        # Clean topic name
        clean = topic.replace(' ', '_')
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{clean}'
        headers = {'User-Agent': 'BookBot-RandomPretrain/1.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('extract', '')
    except Exception:
        pass
    return None


def pretrain():
    print("=" * 60)
    print("  Random Wikipedia Pre-training")
    print("=" * 60)
    
    ai = ConversationalAI()
    
    # Generate all topics
    all_topics = []
    for category, topics in CATEGORIES.items():
        for topic in topics:
            all_topics.append((category, topic))
    
    random.shuffle(all_topics)
    
    total_topics = len(all_topics)
    total_queries = total_topics * len(PATTERNS) + len(INTENTS) * 5
    trained = 0
    errors = 0
    
    print(f"\nCategories: {len(CATEGORIES)}")
    print(f"Topics: {total_topics}")
    print(f"Patterns: {len(PATTERNS)}")
    print(f"Total queries: {total_queries}")
    print("\nTraining on random topics...\n")
    
    # Train on topics
    for i, (category, topic) in enumerate(all_topics):
        # Get Wikipedia summary
        summary = get_wikipedia_summary(topic)
        
        if not summary:
            errors += 1
            continue
        
        # Train on each pattern
        for pattern in PATTERNS:
            query = pattern.format(topic=topic)
            
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
        
        # Progress
        if (i + 1) % 10 == 0:
            progress = (i + 1) / total_topics * 100
            print(f"  [{category}] {topic}: {progress:.0f}% ({i+1}/{total_topics})")
        
        # Rate limit
        time.sleep(0.05)
    
    # Train on intents
    print("\nTraining on intents...")
    for intent, phrases in INTENTS.items():
        for phrase in phrases:
            if ai._intent_classifier:
                ai._intent_classifier.train(phrase, intent, positive=True)
            trained += 1
        print(f"  Intent '{intent}': {len(phrases)} phrases")
    
    # Train on search results
    print("\nTraining on search results...")
    search_topics = ['programming languages', 'famous people', 'world history', 'space exploration',
                     'medical breakthroughs', 'environmental issues', 'cultural movements', 'technological innovations']
    
    for topic in search_topics:
        try:
            url = 'https://en.wikipedia.org/w/api.php'
            params = {'action': 'query', 'list': 'search', 'srsearch': topic, 'format': 'json', 'srlimit': 5}
            headers = {'User-Agent': 'BookBot-RandomPretrain/1.0'}
            resp = requests.get(url, params=params, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                results = [s['title'] for s in data.get('query', {}).get('search', [])]
                for title in results:
                    if ai._neural_mapper:
                        ai._neural_mapper.train(topic, title, positive=True)
                    trained += 1
                print(f"  Search '{topic}': {len(results)} results")
        except Exception:
            pass
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
    
    if ai._response_selector:
        ai._response_selector.save(os.path.join(base, 'response_scores.json'))
        print(f"  response_scores.json: {len(ai._response_selector.response_scores)} preferences")
    
    print("\n" + "=" * 60)
    print(f"  Pre-training complete!")
    print(f"  Trained: {trained} queries")
    print(f"  Topics: {total_topics}")
    print(f"  Errors: {errors}")
    print("=" * 60)


if __name__ == '__main__':
    pretrain()
