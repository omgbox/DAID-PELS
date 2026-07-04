"""Flask app for DAID-PELS web interface."""
import sys, os, time, json, threading, logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, render_template, request, jsonify
import psutil

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = os.urandom(24)

    _chatbot = None
    _stats = {'total_queries': 0, 'total_time': 0, 'avg_response_time': 0, 'start_time': time.time()}
    _history = defaultdict(list)
    _lock = threading.Lock()
    _processing = False
    _cache = {}  # question -> response cache
    _last_query_time = {}  # session_id -> last query time
    _learned_facts = defaultdict(list)  # session_id -> learned facts

    def get_chatbot():
        nonlocal _chatbot
        if _chatbot is None:
            from bookbot.query.conversational_ai import ConversationalAI
            _chatbot = ConversationalAI()
            # Load learned knowledge from database
            try:
                if _chatbot.db:
                    rows = _chatbot.db.execute(
                        "SELECT topic, fact FROM learned_knowledge WHERE source='wikipedia' ORDER BY confidence DESC LIMIT 100"
                    )
                    for row in rows:
                        _learned_facts[row[0]].append(row[1])
                    logger.info(f"Loaded {sum(len(v) for v in _learned_facts.values())} learned facts")
            except Exception as e:
                logger.debug(f"Failed to load learned facts: {e}")
        return _chatbot
    
    def store_wikipedia_fact(question, answer, source):
        """Store Wikipedia answer for future reference."""
        if source != 'wikipedia':
            return
        try:
            bot = get_chatbot()
            # Extract topic from question
            topic = bot._extract_topic(question)
            # Store in learned knowledge
            if hasattr(bot, 'db') and bot.db:
                bot.db.execute(
                    "INSERT OR IGNORE INTO learned_knowledge (topic, fact, source, confidence) VALUES (?, ?, ?, ?)",
                    (topic, answer, 'wikipedia', 0.8)
                )
                bot.db.commit()
            logger.debug(f"Stored Wikipedia fact: {topic}")
        except Exception as e:
            logger.debug(f"Failed to store fact: {e}")

    @app.route('/')
    def index():
        return render_template('chat.html')

    @app.route('/chat', methods=['POST'])
    def chat():
        nonlocal _processing
        data = request.get_json()
        message = data.get('message', '').strip()
        sid = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'response': 'Please enter a message.'})
        
        # Rate limit: max 1 query per 2 seconds per session
        now = time.time()
        if sid in _last_query_time and now - _last_query_time[sid] < 2:
            return jsonify({'response': 'Please wait a moment before asking another question.', 'source': 'local', 'response_time': 0})
        _last_query_time[sid] = now
        
        # Check if already processing
        if _processing:
            return jsonify({'response': 'Still processing previous question...', 'source': 'local', 'response_time': 0})
        
        # Check cache
        cache_key = message.lower().strip()
        if cache_key in _cache:
            cached = _cache[cache_key]
            return jsonify({'response': cached['response'], 'response_time': 0.001, 'source': cached['source']})
        
        _processing = True
        try:
            t = time.time()
            bot = get_chatbot()
            response = bot.chat(message)
            rt = time.time() - t
            
            _stats['total_queries'] += 1
            _stats['total_time'] += rt
            _stats['avg_response_time'] = _stats['total_time'] / _stats['total_queries']
            
            source = 'wikipedia' if 'Wikipedia' in response else 'books' if 'Book database' in response else 'local'
            
            # Cache response (keep last 100 unique queries)
            if len(_cache) > 100:
                oldest = list(_cache.keys())[0]
                del _cache[oldest]
            _cache[cache_key] = {'response': response, 'source': source}
            
            # Store Wikipedia facts for future reference
            store_wikipedia_fact(message, response, source)
            
            _history[sid].append({'user': message, 'bot': response, 'source': source, 'time': round(rt, 3)})
            if len(_history[sid]) > 50:
                _history[sid] = _history[sid][-50:]
            
            return jsonify({'response': response, 'response_time': round(rt, 3), 'source': source})
        except Exception as e:
            return jsonify({'response': str(e), 'source': 'local', 'response_time': 0})
        finally:
            _processing = False

    @app.route('/history')
    def history():
        sid = request.args.get('session_id', 'default')
        return jsonify({'history': _history.get(sid, [])})

    @app.route('/stats')
    def stats():
        bot = get_chatbot()
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        nn = {}
        for name, attr in [('topic_extractor', '_topic_extractor'), ('wiki_mapper', '_neural_mapper'),
                           ('intent_classifier', '_intent_classifier'), ('response_selector', '_response_selector')]:
            obj = getattr(bot, attr, None)
            if obj:
                nn[name] = {'name': name.replace('_', ' ').title(),
                           'architecture': f'{obj.input_dim}→{obj.hidden1}→{obj.hidden2}→1',
                           'weights': f'{obj.input_dim * obj.hidden1 + obj.hidden1 * obj.hidden2 + obj.hidden2:,}',
                           'training_count': getattr(obj, 'training_count', 0)}
        return jsonify({'uptime': time.time() - _stats['start_time'], 'memory_mb': round(mem, 1),
                       'stats': _stats, 'neural_networks': nn, 'cache_size': len(_cache)})

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'processing': _processing})

    return app
