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
    Neural network that learns query → Wikipedia page mappings.
    
    Architecture:
    - Feature extraction (word overlap, n-grams, edit distance)
    - 2-layer feedforward network with 128 hidden neurons
    - Online learning from successful lookups
    """
    
    def __init__(self, input_dim: int = 20, hidden1: int = 512, hidden2: int = 256):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden1)
        scale3 = math.sqrt(2.0 / hidden2)
        
        # Layer 1: input -> hidden1
        self.W1 = [scale1 * (hash(f"w1_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(input_dim) 
                   for j in range(hidden1)]
        self.b1 = [0.0] * hidden1
        
        # Layer 2: hidden1 -> hidden2
        self.W2 = [scale2 * (hash(f"w2_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(hidden1) 
                   for j in range(hidden2)]
        self.b2 = [0.0] * hidden2
        
        # Layer 3: hidden2 -> output
        self.W3 = [scale3 * (hash(f"w3_{j}") % 1000 / 500 - 1) 
                   for j in range(hidden2)]
        self.b3 = [0.0]
        
        # Learned mappings cache
        self.learned_mappings: Dict[str, str] = {}
        
        # Training data
        self.training_history: List[Tuple[str, str, float]] = []
    
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
        
        # 17. Title words in query (proportion)
        if title_words:
            features.append(len([w for w in title_words if w in query_words]) / len(title_words))
        else:
            features.append(0.0)
        
        # 18. Longest common substring length
        lcs_len = self._longest_common_substring(query_lower, title_lower)
        features.append(lcs_len / max(len(query_lower), len(title_lower), 1))
        
        # 19. Soundex similarity
        features.append(1.0 if self._soundex(query_lower.split()[0] if query_lower.split() else '') == 
                       self._soundex(title_lower.split()[0] if title_lower.split() else '') else 0.0)
        
        # 20. Levenshtein similarity per word
        q_tokens = query_lower.split()
        t_tokens = title_lower.split()
        if q_tokens and t_tokens:
            avg_sim = sum(max(self._word_similarity(q, t) for t in t_tokens) for q in q_tokens) / len(q_tokens)
            features.append(avg_sim)
        else:
            features.append(0.0)
        
        return features[:self.input_dim]  # Ensure correct dimension
    
    def _longest_common_substring(self, s1: str, s2: str) -> int:
        """Find length of longest common substring."""
        if not s1 or not s2:
            return 0
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
        return max_len
    
    def _soundex(self, word: str) -> str:
        """Simple Soundex implementation."""
        if not word:
            return ""
        word = word.lower()
        # First letter
        soundex = word[0].upper()
        # Mapping
        mapping = {
            'b': '1', 'f': '1', 'p': '1', 'v': '1',
            'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
            'd': '3', 't': '3',
            'l': '4',
            'm': '5', 'n': '5',
            'r': '6',
        }
        prev = mapping.get(word[0], '0')
        for char in word[1:]:
            code = mapping.get(char, '0')
            if code != '0' and code != prev:
                soundex += code
                if len(soundex) == 4:
                    break
            prev = code
        # Pad with zeros
        soundex = (soundex + '000')[:4]
        return soundex
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        if word1 == word2:
            return 1.0
        # Simple character overlap
        set1 = set(word1)
        set2 = set(word2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
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
        """Forward pass through the 3-layer network."""
        # Ensure correct input size
        x = features[:self.input_dim]
        while len(x) < self.input_dim:
            x.append(0.0)
        
        # Layer 1: input -> hidden1 with ReLU
        hidden1 = []
        for j in range(self.hidden1):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += x[i] * self.W1[i * self.hidden1 + j]
            hidden1.append(max(0, val))  # ReLU
        
        # Layer 2: hidden1 -> hidden2 with ReLU
        hidden2 = []
        for j in range(self.hidden2):
            val = self.b2[j]
            for i in range(self.hidden1):
                val += hidden1[i] * self.W2[i * self.hidden2 + j]
            hidden2.append(max(0, val))  # ReLU
        
        # Layer 3: hidden2 -> output
        output = self.b3[0]
        for j in range(self.hidden2):
            output += hidden2[j] * self.W3[j]
        
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
        
        # Gradient descent (3 layers)
        learning_rate = 0.01
        
        # Update layer 3 weights
        for j in range(self.hidden2):
            self.W3[j] += learning_rate * error * self.b2[j] if j < len(self.b2) else 0
        
        # Update layer 2 weights (simplified)
        for i in range(self.hidden1):
            for j in range(self.hidden2):
                self.W2[i * self.hidden2 + j] += learning_rate * error * 0.01
        
        # Update layer 1 weights (simplified)
        for ii in range(min(len(features), self.input_dim)):
            for jj in range(self.hidden1):
                self.W1[ii * self.hidden1 + jj] += learning_rate * error * features[ii] * 0.01
        
        logger.debug(f"Trained on '{query}' -> '{correct_title}' (error: {error:.3f})")
    
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
            # Merge with existing mappings
            loaded = data.get('learned_mappings', {})
            self.learned_mappings.update(loaded)
            logger.info(f"Loaded {len(loaded)} mappings from {path}, total: {len(self.learned_mappings)}")
        except FileNotFoundError:
            logger.info(f"No saved mappings found at {path}")
