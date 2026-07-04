"""
Flask Application Factory
Creates and configures the Flask app.
"""

import sys
import os
import time
import psutil
from pathlib import Path

# Add parent directory to path (C:\projects)
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, jsonify


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.secret_key = os.urandom(24)
    
    # Lazy-load chatbot
    _chatbot = None
    _stats = {
        'total_queries': 0,
        'total_time': 0,
        'avg_response_time': 0,
        'start_time': time.time(),
    }
    
    def get_chatbot():
        nonlocal _chatbot
        if _chatbot is None:
            # Import via bookbot junction (C:\projects\bookbot -> C:\projects\DAID-PELS)
            from bookbot.query.conversational_ai import ConversationalAI
            _chatbot = ConversationalAI()
        return _chatbot
    
    @app.route('/')
    def index():
        return render_template('chat.html')
    
    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': 'Please enter a message.'})
        
        try:
            start_time = time.time()
            chatbot = get_chatbot()
            response = chatbot.chat(message)
            response_time = time.time() - start_time
            
            # Update stats
            _stats['total_queries'] += 1
            _stats['total_time'] += response_time
            _stats['avg_response_time'] = _stats['total_time'] / _stats['total_queries']
            
            return jsonify({
                'response': response,
                'response_time': round(response_time, 3),
                'stats': _stats
            })
        except Exception as e:
            return jsonify({'response': f'Error: {str(e)}'})
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})
    
    @app.route('/stats')
    def stats():
        chatbot = get_chatbot()
        
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get neural network info
        neural_info = {}
        
        # Topic Extractor
        if hasattr(chatbot, '_topic_extractor') and chatbot._topic_extractor:
            ext = chatbot._topic_extractor
            neural_info['topic_extractor'] = {
                'name': 'Topic Extractor',
                'architecture': f'{ext.input_dim}→{ext.hidden1}→{ext.hidden2}→1',
                'weights': f'{ext.input_dim * ext.hidden1 + ext.hidden1 * ext.hidden2 + ext.hidden2:,}',
                'status': 'loaded',
                'training_count': ext.training_count,
            }
        
        # Wikipedia Mapper
        if hasattr(chatbot, '_neural_mapper') and chatbot._neural_mapper:
            mapper = chatbot._neural_mapper
            neural_info['wiki_mapper'] = {
                'name': 'Wikipedia Mapper',
                'architecture': f'{mapper.input_dim}→{mapper.hidden1}→{mapper.hidden2}→1',
                'weights': f'{mapper.input_dim * mapper.hidden1 + mapper.hidden1 * mapper.hidden2 + mapper.hidden2:,}',
                'status': 'loaded',
                'mappings_count': len(mapper.learned_mappings),
            }
        
        # Intent Classifier
        if hasattr(chatbot, '_intent_classifier') and chatbot._intent_classifier:
            clf = chatbot._intent_classifier
            neural_info['intent_classifier'] = {
                'name': 'Intent Classifier',
                'architecture': f'{clf.input_dim}→{clf.hidden1}→{clf.hidden2}→1',
                'weights': f'{clf.input_dim * clf.hidden1 + clf.hidden1 * clf.hidden2 + clf.hidden2:,}',
                'status': 'loaded',
                'training_count': clf.training_count,
            }
        
        # Response Selector
        if hasattr(chatbot, '_response_selector') and chatbot._response_selector:
            sel = chatbot._response_selector
            neural_info['response_selector'] = {
                'name': 'Response Selector',
                'architecture': f'{sel.input_dim}→{sel.hidden1}→{sel.hidden2}→1',
                'weights': f'{sel.input_dim * sel.hidden1 + sel.hidden1 * sel.hidden2 + sel.hidden2:,}',
                'status': 'loaded',
                'training_count': sel.training_count,
            }
        
        # DistilGPT2
        if hasattr(chatbot, '_generator') and chatbot._generator:
            neural_info['distilgpt2'] = {
                'name': 'DistilGPT2',
                'architecture': 'Transformer (82M params)',
                'weights': '82,000,000',
                'status': 'loaded',
            }
        
        # T5 Paraphrase
        if hasattr(chatbot, '_rewriter') and chatbot._rewriter:
            neural_info['t5_paraphrase'] = {
                'name': 'T5 Paraphrase',
                'architecture': 'Encoder-Decoder (60M params)',
                'weights': '60,000,000',
                'status': 'loaded',
            }
        
        return jsonify({
            'status': 'ok',
            'uptime': round(time.time() - _stats['start_time'], 0),
            'memory_mb': round(memory_info.rss / 1024 / 1024, 1),
            'stats': _stats,
            'neural_networks': neural_info,
        })
    
    return app
