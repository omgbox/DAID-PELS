"""
DAID-PELS Web Interface
Simple Flask app for chatting with the bot.
"""

import sys
import os
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from query.conversational_ai import ConversationalAI

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize the chatbot
chatbot = ConversationalAI()

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
        response = chatbot.chat(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DAID-PELS Web Interface")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
