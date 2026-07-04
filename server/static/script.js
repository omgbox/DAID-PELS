const chat = document.getElementById('chat');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');
const statsPanel = document.getElementById('statsPanel');
const statsContent = document.getElementById('statsContent');
const snackbar = document.getElementById('snackbar');
const welcome = document.getElementById('welcome');

// Generate unique session ID
const sessionId = localStorage.getItem('chat_session_id') || generateSessionId();
localStorage.setItem('chat_session_id', sessionId);

// State
let lastStats = null;
let statsInterval = null;

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Send on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Focus input on load
messageInput.focus();

// Load chat history on page load
loadHistory();

// Load stats on page load
loadStats();

// Auto-refresh stats every 2 seconds when panel is open
setInterval(() => {
    if (statsPanel.classList.contains('show')) {
        loadStats();
    }
}, 2000);

function toggleStats() {
    statsPanel.classList.toggle('show');
    if (statsPanel.classList.contains('show')) {
        loadStats();
        statsInterval = setInterval(loadStats, 1000);
    } else {
        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }
    }
}

// Snackbar notification
function showSnackbar(source, text) {
    const icons = {
        'wikipedia': '🌐',
        'books': '📚',
        'local': '💬',
        'thinking': '🧠'
    };
    
    const colors = {
        'wikipedia': '#4ecca3',
        'books': '#e94560',
        'local': '#0f3460',
        'thinking': '#666'
    };
    
    snackbar.querySelector('.snackbar-icon').textContent = icons[source] || '💬';
    snackbar.querySelector('.snackbar-text').textContent = text || getSourceText(source);
    snackbar.style.background = colors[source] || '#0f3460';
    snackbar.classList.add('show');
    
    setTimeout(() => {
        snackbar.classList.remove('show');
    }, 2000);
}

function getSourceText(source) {
    const texts = {
        'wikipedia': 'Searching Wikipedia...',
        'books': 'Checking book database...',
        'local': 'Thinking...',
        'thinking': 'Processing...'
    };
    return texts[source] || 'Thinking...';
}

// Load chat history
async function loadHistory() {
    try {
        const response = await fetch(`/history?session_id=${sessionId}`);
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            // Hide welcome message
            welcome.style.display = 'none';
            
            // Load all messages
            data.history.forEach(msg => {
                addMessageToChat(msg.user, 'user');
                addMessageToChat(msg.bot, 'bot', msg.time, msg.source);
            });
        }
    } catch (error) {
        console.log('Failed to load history');
    }
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (!lastStats || JSON.stringify(data) !== JSON.stringify(lastStats)) {
            renderStats(data);
            lastStats = data;
        }
    } catch (error) {
        statsContent.innerHTML = '<div class="loading">Failed to load stats</div>';
    }
}

function renderStats(data) {
    const uptime = formatUptime(data.uptime);
    const avgTime = data.stats.avg_response_time;
    const maxTime = 5;
    const progress = Math.min((avgTime / maxTime) * 100, 100);
    
    let html = `
        <div class="stat-section">
            <h4>System</h4>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-label">Uptime</div>
                    <div class="stat-value">${uptime}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Memory</div>
                    <div class="stat-value">${data.memory_mb} MB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Total Queries</div>
                    <div class="stat-value" id="stat-queries">${data.stats.total_queries}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Avg Response</div>
                    <div class="stat-value" id="stat-avg-time">${avgTime.toFixed(2)}s</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="stat-section">
            <h4>Neural Networks</h4>
    `;
    
    for (const [key, nn] of Object.entries(data.neural_networks)) {
        const nnProgress = nn.training_count ? Math.min((nn.training_count / 100) * 100, 100) : 0;
        html += `
            <div class="neural-network">
                <div class="nn-header">
                    <span class="nn-name">${nn.name}</span>
                    <span class="nn-status loaded">${nn.status}</span>
                </div>
                <div class="nn-details">
                    <span>Architecture: ${nn.architecture}</span>
                    <span>Weights: ${nn.weights}</span>
                    ${nn.training_count !== undefined ? `<span>Trained: ${nn.training_count}x</span>` : ''}
                    ${nn.mappings_count !== undefined ? `<span>Mappings: ${nn.mappings_count}</span>` : ''}
                </div>
                ${nn.training_count !== undefined ? `
                <div class="progress-bar">
                    <div class="progress-fill neural" style="width: ${nnProgress}%"></div>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    html += '</div>';
    statsContent.innerHTML = html;
}

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

function sendSuggestion(text) {
    messageInput.value = text;
    sendMessage();
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Hide welcome
    welcome.style.display = 'none';

    // Add user message
    addMessageToChat(message, 'user');
    messageInput.value = '';
    sendBtn.disabled = true;

    // Show typing indicator
    typing.classList.add('show');
    showSnackbar('thinking', 'Processing your message...');
    chat.scrollTop = chat.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId })
        });

        const data = await response.json();
        
        // Hide typing
        typing.classList.remove('show');
        
        // Show source snackbar
        showSnackbar(data.source, `Answer from ${data.source === 'wikipedia' ? 'Wikipedia' : data.source === 'books' ? 'Book Database' : 'Local Knowledge'}`);
        
        // Add bot response
        addMessageToChat(data.response, 'bot', data.response_time, data.source);
        
        // Update stats
        if (statsPanel.classList.contains('show')) {
            loadStats();
        }
    } catch (error) {
        typing.classList.remove('show');
        showSnackbar('local', 'Error occurred');
        addMessageToChat('Sorry, something went wrong. Please try again.', 'bot');
    }

    sendBtn.disabled = false;
    messageInput.focus();
}

function addMessageToChat(text, type, responseTime = null, source = null) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    
    let content = '';
    
    // Parse source attribution
    if (type === 'bot' && text.includes('— Source:')) {
        const parts = text.split('— Source:');
        content = `
            <div>${parts[0].trim()}</div>
            <div class="source">— Source:${parts[1]}</div>
        `;
    } else {
        content = text;
    }
    
    // Add response time for bot messages
    if (type === 'bot' && responseTime !== null) {
        content += `<div class="response-time">${responseTime.toFixed(2)}s</div>`;
    }
    
    // Add source badge
    if (type === 'bot' && source) {
        const sourceBadge = source === 'wikipedia' ? '🌐 Wikipedia' : 
                           source === 'books' ? '📚 Books' : '💬 Local';
        content += `<div class="source-badge">${sourceBadge}</div>`;
    }
    
    div.innerHTML = content;
    
    // Insert before typing indicator
    chat.insertBefore(div, typing);
    chat.scrollTop = chat.scrollHeight;
}
