"""
Neural Topic Extractor
Learns to extract key topics from any query using a small neural network.
Replaces regex patterns with learned behavior.
"""

import re
import math
import logging
from typing import List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger('bookbot.neural_topic_extractor')


class NeuralTopicExtractor:
    """
    Small neural network that extracts topics from queries.
    
    Architecture:
    - Feature extraction per word (position, length, case, etc.)
    - 2-layer feedforward network
    - Online learning from successful lookups
    """
    
    # Common stop words
    STOP_WORDS = {
        'what', 'who', 'when', 'where', 'why', 'how', 'which',
        'is', 'are', 'was', 'were', 'do', 'does', 'did',
        'have', 'has', 'had', 'can', 'could', 'will', 'would', 'should',
        'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to',
        'tell', 'me', 'about', 'you', 'please', 'i', 'my', 'your',
        'this', 'that', 'these', 'those', 'it', 'its',
    }
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights with Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        
        self.W1 = [scale1 * (hash(f"t1_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(input_dim) 
                   for j in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        
        self.W2 = [scale2 * (hash(f"t2_{j}") % 1000 / 500 - 1) 
                   for j in range(hidden_dim)]
        self.b2 = [0.0]
        
        # Learned word importance scores
        self.word_scores: dict = {}
        
        # Training history
        self.training_count = 0
    
    def _extract_word_features(self, word: str, position: int, 
                               total_words: int, query: str) -> List[float]:
        """Extract features for a single word."""
        features = []
        
        # 1. Position normalized (beginning, middle, end)
        features.append(position / max(total_words - 1, 1))
        
        # 2. Word length normalized
        features.append(min(len(word) / 10, 1.0))
        
        # 3. Is capitalized (proper noun hint)
        features.append(1.0 if word[0].isupper() and position > 0 else 0.0)
        
        # 4. Is all caps (abbreviation hint)
        features.append(1.0 if word.isupper() and len(word) > 1 else 0.0)
        
        # 5. Is stop word
        features.append(1.0 if word.lower() in self.STOP_WORDS else 0.0)
        
        # 6. Contains vowels (likely a real word)
        features.append(1.0 if any(c in word.lower() for c in 'aeiou') else 0.0)
        
        # 7. Is short (1-2 chars, likely not important)
        features.append(1.0 if len(word) <= 2 else 0.0)
        
        # 8. Is long (5+ chars, likely important)
        features.append(1.0 if len(word) >= 5 else 0.0)
        
        # 9. Is first word
        features.append(1.0 if position == 0 else 0.0)
        
        # 10. Is last word
        features.append(1.0 if position == total_words - 1 else 0.0)
        
        # 11. Previous word is stop word (subject hint)
        if position > 0:
            prev_word = query.split()[position - 1] if position <= len(query.split()) else ''
            features.append(1.0 if prev_word.lower() in self.STOP_WORDS else 0.0)
        else:
            features.append(0.0)
        
        # 12. Next word is stop word (object hint)
        if position < total_words - 1:
            next_word = query.split()[position + 1] if position + 1 < len(query.split()) else ''
            features.append(1.0 if next_word.lower() in self.STOP_WORDS else 0.0)
        else:
            features.append(0.0)
        
        # 13. Learned word score
        features.append(self.word_scores.get(word.lower(), 0.5))
        
        # 14. Word appears in title case in original query
        original_words = re.findall(r'\w+', query)
        if position < len(original_words):
            features.append(1.0 if original_words[position][0].isupper() else 0.0)
        else:
            features.append(0.0)
        
        # 15. Is question word
        question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        features.append(1.0 if word.lower() in question_words else 0.0)
        
        # 16. Is verb-like (contains common verb patterns)
        verb_suffixes = {'ed', 'ing', 'es', 's'}
        features.append(1.0 if any(word.lower().endswith(s) for s in verb_suffixes) else 0.0)
        
        return features[:self.input_dim]
    
    def _forward(self, features: List[float]) -> float:
        """Forward pass through the network."""
        x = features[:self.input_dim]
        while len(x) < self.input_dim:
            x.append(0.0)
        
        # Hidden layer with ReLU
        hidden = []
        for j in range(self.hidden_dim):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += x[i] * self.W1[i * self.hidden_dim + j]
            hidden.append(max(0, val))  # ReLU
        
        # Output layer (single score)
        output = self.b2[0]
        for j in range(self.hidden_dim):
            output += hidden[j] * self.W2[j]
        
        # Sigmoid to get probability
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, output))))
    
    def extract_topic(self, query: str) -> str:
        """
        Extract the main topic from a query using neural scoring.
        
        Args:
            query: The user's query
            
        Returns:
            Extracted topic string
        """
        # Tokenize
        words = re.findall(r'\w+', query)
        if not words:
            return query
        
        # Score each word
        word_scores = []
        for i, word in enumerate(words):
            features = self._extract_word_features(word, i, len(words), query)
            score = self._forward(features)
            word_scores.append((score, word, i))
        
        # Sort by score
        word_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Select top words (at least 1, up to 4)
        # Filter out very low scoring words
        threshold = 0.3
        selected = []
        for score, word, pos in word_scores:
            if score >= threshold and len(selected) < 4:
                selected.append((pos, word))
            elif len(selected) >= 2:
                break
        
        # If no words selected, take top 2
        if not selected:
            selected = [(word_scores[0][2], word_scores[0][1])]
            if len(word_scores) > 1:
                selected.append((word_scores[1][2], word_scores[1][1]))
        
        # Sort by position to maintain order
        selected.sort(key=lambda x: x[0])
        
        # Filter out stop words
        filtered = [(pos, word) for pos, word in selected 
                    if word.lower() not in self.STOP_WORDS]
        
        # If all filtered, keep at least the first non-stop word
        if not filtered and selected:
            for pos, word in selected:
                if word.lower() not in self.STOP_WORDS:
                    filtered = [(pos, word)]
                    break
        
        # If still empty, use top scored word regardless
        if not filtered:
            filtered = [(word_scores[0][2], word_scores[0][1])]
        
        # Combine into topic
        topic = ' '.join(word for _, word in filtered)
        
        return topic.lower()
    
    def train(self, query: str, correct_topic: str, positive: bool = True):
        """
        Train on a lookup result.
        
        Args:
            query: The query that was asked
            correct_topic: The topic that worked
            positive: True if this was correct
        """
        words = re.findall(r'\w+', query)
        correct_words = set(correct_topic.lower().split())
        
        # Update word scores
        for word in words:
            word_lower = word.lower()
            if word_lower in correct_words:
                # Boost score for words in correct topic
                current = self.word_scores.get(word_lower, 0.5)
                self.word_scores[word_lower] = min(1.0, current + 0.1)
            elif positive:
                # Slightly penalize words not in correct topic
                current = self.word_scores.get(word_lower, 0.5)
                self.word_scores[word_lower] = max(0.0, current - 0.05)
        
        # Train neural network on each word
        for i, word in enumerate(words):
            features = self._extract_word_features(word, i, len(words), query)
            prediction = self._forward(features)
            target = 1.0 if word.lower() in correct_words else 0.0
            error = target - prediction
            
            # Simple gradient descent
            learning_rate = 0.01
            for j in range(self.hidden_dim):
                self.W2[j] += learning_rate * error * features[j] if j < len(features) else 0
            
            for ii in range(min(len(features), self.input_dim)):
                for jj in range(self.hidden_dim):
                    self.W1[ii * self.hidden_dim + jj] += learning_rate * error * features[ii] * 0.1
        
        self.training_count += 1
        logger.debug(f"Trained topic extractor on '{query}' -> '{correct_topic}'")
    
    def save(self, path: str):
        """Save learned word scores to file."""
        import json
        data = {
            'word_scores': self.word_scores,
            'training_count': self.training_count,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.word_scores)} word scores to {path}")
    
    def load(self, path: str):
        """Load learned word scores from file."""
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.word_scores = data.get('word_scores', {})
            self.training_count = data.get('training_count', 0)
            logger.info(f"Loaded {len(self.word_scores)} word scores from {path}")
        except FileNotFoundError:
            logger.info(f"No saved word scores found at {path}")
