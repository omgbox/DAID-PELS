"""
Neural Intent Classifier
Classifies user messages into intents using a 3-layer neural network.
Replaces rule-based regex patterns.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger('bookbot.neural_intent_classifier')


class NeuralIntentClassifier:
    """
    3-layer neural network that classifies user intents.
    
    Architecture:
    - Input: word features (position, length, type, etc.)
    - Hidden1: 128 neurons
    - Hidden2: 64 neurons  
    - Output: intent probability
    """
    
    # Intent types
    INTENTS = {
        'greeting': 0,
        'farewell': 1,
        'question': 2,
        'statement': 3,
        'personal': 4,
        'emotional': 5,
        'command': 6,
        'book_query': 7,
    }
    
    # Common words for each intent
    INTENT_WORDS = {
        'greeting': {'hi', 'hello', 'hey', 'howdy', 'sup', 'yo', 'greetings', 'morning', 'afternoon', 'evening'},
        'farewell': {'bye', 'goodbye', 'see', 'later', 'take', 'care', 'night', 'quit', 'exit'},
        'personal': {'i', 'my', 'me', 'mine', 'myself', 'am', "i'm", 'like', 'love', 'hate', 'work', 'live', 'name'},
        'emotional': {'feel', 'feeling', 'happy', 'sad', 'angry', 'excited', 'tired', 'stressed', 'great', 'terrible'},
        'command': {'show', 'tell', 'give', 'find', 'search', 'look', 'help', 'explain'},
    }
    
    def __init__(self, input_dim: int = 20, hidden1: int = 256, hidden2: int = 128):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden1)
        scale3 = math.sqrt(2.0 / hidden2)
        
        # Layer 1
        self.W1 = [scale1 * (hash(f"ic1_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(input_dim) 
                   for j in range(hidden1)]
        self.b1 = [0.0] * hidden1
        
        # Layer 2
        self.W2 = [scale2 * (hash(f"ic2_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(hidden1) 
                   for j in range(hidden2)]
        self.b2 = [0.0] * hidden2
        
        # Layer 3 (output)
        self.W3 = [scale3 * (hash(f"ic3_{j}") % 1000 / 500 - 1) 
                   for j in range(hidden2)]
        self.b3 = [0.0]
        
        # Learned intent scores
        self.intent_scores: Dict[str, float] = {}
        
        # Training count
        self.training_count = 0
    
    def _extract_features(self, message: str) -> List[float]:
        """Extract features from message for intent classification."""
        features = []
        m = message.lower().strip()
        words = m.split()
        
        # 1. Message length (normalized)
        features.append(min(len(m) / 100, 1.0))
        
        # 2. Word count (normalized)
        features.append(min(len(words) / 10, 1.0))
        
        # 3. Starts with question word
        question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would'}
        features.append(1.0 if words and words[0] in question_words else 0.0)
        
        # 4. Ends with question mark
        features.append(1.0 if m.endswith('?') else 0.0)
        
        # 5. Starts with I/my (personal)
        features.append(1.0 if words and words[0] in {'i', "i'm", 'my', 'me'} else 0.0)
        
        # 6. Contains greeting
        features.append(1.0 if any(w in self.INTENT_WORDS['greeting'] for w in words) else 0.0)
        
        # 7. Contains farewell
        features.append(1.0 if any(w in self.INTENT_WORDS['farewell'] for w in words) else 0.0)
        
        # 8. Contains emotional word
        features.append(1.0 if any(w in self.INTENT_WORDS['emotional'] for w in words) else 0.0)
        
        # 9. Contains command word
        features.append(1.0 if any(w in self.INTENT_WORDS['command'] for w in words) else 0.0)
        
        # 10. Contains book-related word
        book_words = {'book', 'novel', 'chapter', 'page', 'read', 'story', 'character', 'author'}
        features.append(1.0 if any(w in book_words for w in words) else 0.0)
        
        # 11. Average word length
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        features.append(min(avg_len / 5, 1.0))
        
        # 12. Has exclamation
        features.append(1.0 if '!' in m else 0.0)
        
        # 13. Is very short (< 5 words)
        features.append(1.0 if len(words) < 5 else 0.0)
        
        # 14. Is long (> 15 words)
        features.append(1.0 if len(words) > 15 else 0.0)
        
        # 15. Contains pronoun
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        features.append(1.0 if any(w in pronouns for w in words) else 0.0)
        
        # 16. Contains verb-like word
        verb_patterns = {'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 
                        'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        features.append(1.0 if any(w in verb_patterns for w in words) else 0.0)
        
        # 17. Contains negation
        negations = {'not', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", 'no', 'never'}
        features.append(1.0 if any(w in negations for w in words) else 0.0)
        
        # 18. Starts with tell/show/explain
        command_starts = {'tell', 'show', 'explain', 'give', 'find', 'search', 'look'}
        features.append(1.0 if words and words[0] in command_starts else 0.0)
        
        # 19. Learned intent score
        features.append(self.intent_scores.get(m, 0.5))
        
        # 20. Word overlap with common intents
        overlap_count = 0
        for intent_words in self.INTENT_WORDS.values():
            overlap_count += len(set(words) & intent_words)
        features.append(min(overlap_count / 5, 1.0))
        
        return features[:self.input_dim]
    
    def _forward(self, features: List[float]) -> float:
        """Forward pass through 3-layer network."""
        x = features[:self.input_dim]
        while len(x) < self.input_dim:
            x.append(0.0)
        
        # Layer 1: input -> hidden1 with ReLU
        hidden1 = []
        for j in range(self.hidden1):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += x[i] * self.W1[i * self.hidden1 + j]
            hidden1.append(max(0, val))
        
        # Layer 2: hidden1 -> hidden2 with ReLU
        hidden2 = []
        for j in range(self.hidden2):
            val = self.b2[j]
            for i in range(self.hidden1):
                val += hidden1[i] * self.W2[i * self.hidden2 + j]
            hidden2.append(max(0, val))
        
        # Layer 3: hidden2 -> output
        output = self.b3[0]
        for j in range(self.hidden2):
            output += hidden2[j] * self.W3[j]
        
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, output))))
    
    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify the intent of a message.
        
        Returns:
            (intent_type, confidence)
        """
        m = message.lower().strip()
        words = m.split()
        
        # Quick checks for obvious intents
        if not words:
            return 'statement', 0.5
        
        # Greeting check
        if any(w in self.INTENT_WORDS['greeting'] for w in words):
            return 'greeting', 0.9
        
        # Farewell check
        if any(w in self.INTENT_WORDS['farewell'] for w in words):
            return 'farewell', 0.9
        
        # Personal statement check
        if words[0] in {'i', "i'm", 'my', 'me', 'mine'}:
            return 'personal', 0.8
        
        # Emotional check
        if any(w in self.INTENT_WORDS['emotional'] for w in words):
            return 'emotional', 0.8
        
        # Question check
        if m.endswith('?') or words[0] in {'what', 'who', 'when', 'where', 'why', 'how', 'which'}:
            return 'question', 0.85
        
        # Command check
        if words[0] in self.INTENT_WORDS['command']:
            return 'command', 0.8
        
        # Neural classification for ambiguous cases
        features = self._extract_features(m)
        score = self._forward(features)
        
        # Map score to intent
        if score > 0.7:
            return 'question', score
        elif score > 0.4:
            return 'statement', score
        else:
            return 'statement', 1.0 - score
    
    def train(self, message: str, correct_intent: str, positive: bool = True):
        """Train on a classified message."""
        m = message.lower().strip()
        
        # Update learned scores
        if positive:
            current = self.intent_scores.get(m, 0.5)
            self.intent_scores[m] = min(1.0, current + 0.1)
        
        # Simple online learning
        features = self._extract_features(m)
        prediction = self._forward(features)
        target = 1.0 if positive else 0.0
        error = target - prediction
        
        learning_rate = 0.01
        
        # Update weights
        for j in range(self.hidden2):
            self.W3[j] += learning_rate * error * self.b2[j] if j < len(self.b2) else 0
        
        self.training_count += 1
    
    def save(self, path: str):
        """Save learned scores."""
        import json
        data = {
            'intent_scores': self.intent_scores,
            'training_count': self.training_count,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """Load learned scores."""
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.intent_scores = data.get('intent_scores', {})
            self.training_count = data.get('training_count', 0)
        except FileNotFoundError:
            pass
