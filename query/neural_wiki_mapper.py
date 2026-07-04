"""
Neural Wikipedia Mapper
Learns to map queries to Wikipedia pages using a small neural network.
Trains on successful lookups, improves over time.
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger('bookbot.neural_wiki_mapper')


class NeuralWikipediaMapper:
    """
    Small neural network that learns query → Wikipedia page mappings.
    
    Architecture:
    - Feature extraction (word overlap, n-grams, edit distance)
    - 2-layer feedforward network
    - Online learning from successful lookups
    """
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple weights (no PyTorch needed for this small network)
        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        
        self.W1 = [scale1 * (hash(f"w1_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(input_dim) 
                   for j in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        
        self.W2 = [scale2 * (hash(f"w2_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(hidden_dim) 
                   for j in range(1)]
        self.b2 = [0.0]
        
        # Learned mappings cache
        self.learned_mappings: Dict[str, str] = {}
        
        # Training data
        self.training_history: List[Tuple[str, str, float]] = []
        
        # Feature statistics for normalization
        self.feature_stats = {
            'word_overlap_max': 1.0,
            'char_ngram_max': 1.0,
            'edit_dist_max': 1.0,
            'length_ratio_max': 2.0,
        }
    
    def _extract_features(self, query: str, title: str) -> List[float]:
        """Extract feature vector from query-title pair."""
        query_lower = query.lower().strip()
        title_lower = title.lower().strip()
        
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        
        features = []
        
        # 1. Word overlap (Jaccard-like)
        if query_words and title_words:
            overlap = len(query_words & title_words)
            union = len(query_words | title_words)
            features.append(overlap / max(union, 1))
        else:
            features.append(0.0)
        
        # 2. Query contains title
        features.append(1.0 if title_lower in query_lower else 0.0)
        
        # 3. Title contains query
        features.append(1.0 if query_lower in title_lower else 0.0)
        
        # 4. Character n-gram similarity (3-grams)
        q_ngrams = self._get_ngrams(query_lower, 3)
        t_ngrams = self._get_ngrams(title_lower, 3)
        if q_ngrams and t_ngrams:
            ngram_overlap = len(q_ngrams & t_ngrams)
            ngram_union = len(q_ngrams | t_ngrams)
            features.append(ngram_overlap / max(ngram_union, 1))
        else:
            features.append(0.0)
        
        # 5. Edit distance (normalized)
        edit_dist = self._edit_distance(query_lower, title_lower)
        max_len = max(len(query_lower), len(title_lower), 1)
        features.append(1.0 - edit_dist / max_len)
        
        # 6. Length ratio
        features.append(min(len(query_lower), len(title_lower)) / 
                       max(len(query_lower), len(title_lower), 1))
        
        # 7. First word match
        q_first = query_lower.split()[0] if query_lower.split() else ''
        t_first = title_lower.split()[0] if title_lower.split() else ''
        features.append(1.0 if q_first == t_first else 0.0)
        
        # 8. Last word match
        q_last = query_lower.split()[-1] if query_lower.split() else ''
        t_last = title_lower.split()[-1] if title_lower.split() else ''
        features.append(1.0 if q_last == t_last else 0.0)
        
        # 9. Word count difference
        features.append(abs(len(query_words) - len(title_words)) / 
                       max(len(query_words) + len(title_words), 1))
        
        # 10. Contains parentheses (disambiguation hint)
        features.append(1.0 if '(' in title_lower else 0.0)
        
        # 11. Title starts with query
        features.append(1.0 if title_lower.startswith(query_lower) else 0.0)
        
        # 12. Query starts with title
        features.append(1.0 if query_lower.startswith(title_lower) else 0.0)
        
        # 13. Common words overlap
        common_words = {'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to', 'is', 'are'}
        q_common = query_words - common_words
        t_common = title_words - common_words
        if q_common and t_common:
            features.append(len(q_common & t_common) / max(len(q_common | t_common), 1))
        else:
            features.append(0.0)
        
        # 14. Character overlap ratio
        q_chars = set(query_lower.replace(' ', ''))
        t_chars = set(title_lower.replace(' ', ''))
        if q_chars and t_chars:
            features.append(len(q_chars & t_chars) / max(len(q_chars | t_chars), 1))
        else:
            features.append(0.0)
        
        # 15. Exact match after removing disambiguation
        q_clean = re.sub(r'\s*\(.*?\)\s*', '', query_lower).strip()
        t_clean = re.sub(r'\s*\(.*?\)\s*', '', title_lower).strip()
        features.append(1.0 if q_clean == t_clean else 0.0)
        
        # 16. Query words in title (proportion)
        if query_words:
            features.append(len([w for w in query_words if w in title_words]) / len(query_words))
        else:
            features.append(0.0)
        
        return features[:self.input_dim]  # Ensure correct dimension
    
    def _get_ngrams(self, text: str, n: int) -> set:
        """Get character n-grams."""
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def _forward(self, features: List[float]) -> float:
        """Forward pass through the network."""
        # Ensure correct input size
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
    
    def predict(self, query: str, candidates: List[str]) -> Optional[str]:
        """
        Predict the best Wikipedia page for a query.
        
        Args:
            query: The user's query
            candidates: List of Wikipedia page titles to choose from
        
        Returns:
            Best matching page title, or None
        """
        if not candidates:
            return None
        
        # Check learned mappings first
        query_lower = query.lower().strip()
        if query_lower in self.learned_mappings:
            mapped = self.learned_mappings[query_lower]
            if mapped in candidates:
                return mapped
        
        # Score each candidate
        best_score = -1
        best_title = None
        
        for title in candidates:
            features = self._extract_features(query, title)
            score = self._forward(features)
            
            if score > best_score:
                best_score = score
                best_title = title
        
        # Only return if confidence is reasonable
        if best_score > 0.3:
            return best_title
        
        return None
    
    def train(self, query: str, correct_title: str, positive: bool = True):
        """
        Train on a lookup result.
        
        Args:
            query: The query that was asked
            correct_title: The Wikipedia page that was correct
            positive: True if this was a correct mapping
        """
        query_lower = query.lower().strip()
        
        # Store the mapping
        if positive:
            self.learned_mappings[query_lower] = correct_title
        
        # Add to training history
        self.training_history.append((query, correct_title, 1.0 if positive else 0.0))
        
        # Simple online learning: adjust weights based on error
        features = self._extract_features(query, correct_title)
        prediction = self._forward(features)
        target = 1.0 if positive else 0.0
        error = target - prediction
        
        # Gradient descent (very simple)
        learning_rate = 0.01
        
        # Update output layer
        for j in range(self.hidden_dim):
            self.W2[j] += learning_rate * error * features[j] if j < len(features) else 0
        
        # Update hidden layer (simplified)
        for i in range(min(len(features), self.input_dim)):
            for j in range(self.hidden_dim):
                self.W1[i * self.hidden_dim + j] += learning_rate * error * features[i] * 0.1
        
        logger.debug(f"Trained on '{query}' → '{correct_title}' (error: {error:.3f})")
    
    def save(self, path: str):
        """Save learned mappings to file."""
        import json
        data = {
            'learned_mappings': self.learned_mappings,
            'training_count': len(self.training_history),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.learned_mappings)} mappings to {path}")
    
    def load(self, path: str):
        """Load learned mappings from file."""
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.learned_mappings = data.get('learned_mappings', {})
            logger.info(f"Loaded {len(self.learned_mappings)} mappings from {path}")
        except FileNotFoundError:
            logger.info(f"No saved mappings found at {path}")
