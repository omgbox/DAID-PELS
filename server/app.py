"""
Flask Application Factory
Creates and configures the Flask app.
"""

import sys
import os
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
            chatbot = get_chatbot()
            response = chatbot.chat(message)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'response': f'Error: {str(e)}'})
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})
    
    return app
