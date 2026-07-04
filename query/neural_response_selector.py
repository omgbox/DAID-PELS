"""
Neural Response Selector
Picks the best response from candidates using a 3-layer neural network.
Replaces random.choice with learned selection.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger('bookbot.neural_response_selector')


class NeuralResponseSelector:
    """
    3-layer neural network that selects the best response.
    
    Architecture:
    - Input: response features (length, fluency, relevance, etc.)
    - Hidden1: 64 neurons
    - Hidden2: 32 neurons
    - Output: quality score
    """
    
    def __init__(self, input_dim: int = 16, hidden1: int = 128, hidden2: int = 64):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden1)
        scale3 = math.sqrt(2.0 / hidden2)
        
        # Layer 1
        self.W1 = [scale1 * (hash(f"rs1_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(input_dim) 
                   for j in range(hidden1)]
        self.b1 = [0.0] * hidden1
        
        # Layer 2
        self.W2 = [scale2 * (hash(f"rs2_{i}_{j}") % 1000 / 500 - 1) 
                   for i in range(hidden1) 
                   for j in range(hidden2)]
        self.b2 = [0.0] * hidden2
        
        # Layer 3
        self.W3 = [scale3 * (hash(f"rs3_{j}") % 1000 / 500 - 1) 
                   for j in range(hidden2)]
        self.b3 = [0.0]
        
        # Learned response preferences
        self.response_scores: Dict[str, float] = {}
        
        # Training count
        self.training_count = 0
    
    def _extract_features(self, response: str, query: str = "") -> List[float]:
        """Extract features from a response."""
        features = []
        r = response.strip()
        words = r.split()
        
        # 1. Response length (normalized)
        features.append(min(len(r) / 500, 1.0))
        
        # 2. Word count (normalized)
        features.append(min(len(words) / 50, 1.0))
        
        # 3. Average word length
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        features.append(min(avg_len / 6, 1.0))
        
        # 4. Has proper sentence structure (starts with capital)
        features.append(1.0 if r and r[0].isupper() else 0.0)
        
        # 5. Ends with punctuation
        features.append(1.0 if r and r[-1] in '.!?' else 0.0)
        
        # 6. Has multiple sentences
        features.append(1.0 if '. ' in r or '!' in r or '?' in r else 0.0)
        
        # 7. Query word overlap
        if query:
            query_words = set(query.lower().split())
            response_words = set(r.lower().split())
            overlap = len(query_words & response_words)
            features.append(min(overlap / max(len(query_words), 1), 1.0))
        else:
            features.append(0.0)
        
        # 8. Has numbers (factual)
        features.append(1.0 if re.search(r'\d', r) else 0.0)
    
    def _forward(self, features: List[float]) -> float:
        """Forward pass through 3-layer network."""
        x = features[:self.input_dim]
        while len(x) < self.input_dim:
            x.append(0.0)
        
        # Layer 1
        hidden1 = []
        for j in range(self.hidden1):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += x[i] * self.W1[i * self.hidden1 + j]
            hidden1.append(max(0, val))
        
        # Layer 2
        hidden2 = []
        for j in range(self.hidden2):
            val = self.b2[j]
            for i in range(self.hidden1):
                val += hidden1[i] * self.W2[i * self.hidden2 + j]
            hidden2.append(max(0, val))
        
        # Layer 3
        output = self.b3[0]
        for j in range(self.hidden2):
            output += hidden2[j] * self.W3[j]
        
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, output))))
    
    def select(self, candidates: List[str], query: str = "") -> str:
        """
        Select the best response from candidates.
        
        Args:
            candidates: List of response options
            query: Original user query for context
            
        Returns:
            Best response string
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate
        scored = []
        for response in candidates:
            # Check learned preference
            cache_key = response[:100].lower()
            if cache_key in self.response_scores:
                score = self.response_scores[cache_key]
            else:
                features = self._extract_features(response, query)
                score = self._forward(features)
            scored.append((score, response))
        
        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return scored[0][1]
    
    def train(self, response: str, query: str = "", positive: bool = True):
        """Train on a response selection."""
        cache_key = response[:100].lower()
        
        # Update learned score
        current = self.response_scores.get(cache_key, 0.5)
        if positive:
            self.response_scores[cache_key] = min(1.0, current + 0.1)
        else:
            self.response_scores[cache_key] = max(0.0, current - 0.1)
        
        # Simple online learning
        features = self._extract_features(response, query)
        prediction = self._forward(features)
        target = 1.0 if positive else 0.0
        error = target - prediction
        
        learning_rate = 0.01
        
        # Update weights
        for j in range(self.hidden2):
            self.W3[j] += learning_rate * error * self.b2[j] if j < len(self.b2) else 0
        
        self.training_count += 1
    
    def save(self, path: str):
        """Save learned preferences."""
        import json
        data = {
            'response_scores': self.response_scores,
            'training_count': self.training_count,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """Load learned preferences."""
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.response_scores = data.get('response_scores', {})
            self.training_count = data.get('training_count', 0)
        except FileNotFoundError:
            pass
