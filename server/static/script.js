const chat = document.getElementById('chat');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');
const statsPanel = document.getElementById('statsPanel');
const statsContent = document.getElementById('statsContent');
const snackbar = document.getElementById('snackbar');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const sidebar = document.getElementById('sidebar');
const sessionsList = document.getElementById('sessionsList');

// Generate unique session ID
let currentSession = localStorage.getItem('chat_session_id') || generateSessionId();
localStorage.setItem('chat_session_id', currentSession);

// State
let lastStats = null;
let statsInterval = null;
let sessions = JSON.parse(localStorage.getItem('chat_sessions') || '[]');

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing...');
    
    // Auto-resize textarea
    messageInput.addEventListener('input', autoResize);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Focus input
    messageInput.focus();
    
    // Load data
    loadHistory();
    loadStats();
    renderSessions();
    
    // Auto-refresh stats
    setInterval(() => {
        if (statsPanel.classList.contains('show')) {
            loadStats();
        }
    }, 2000);
    
    console.log('Initialization complete');
});

function autoResize() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
}

// Sidebar
function toggleSidebar() {
    sidebar.classList.toggle('show');
}

// Sessions
function newSession() {
    currentSession = generateSessionId();
    localStorage.setItem('chat_session_id', currentSession);
    
    // Add to sessions list
    sessions.unshift({
        id: currentSession,
        name: 'New Chat',
        model: 'Wikipedia + Local',
        time: new Date().toLocaleTimeString()
    });
    
    if (sessions.length > 10) sessions = sessions.slice(0, 10);
    localStorage.setItem('chat_sessions', JSON.stringify(sessions));
    
    // Clear messages
    messagesContainer.innerHTML = '<div class="typing" id="typing"><span></span><span></span><span></span></div>';
    welcomeScreen.style.display = 'flex';
    messagesContainer.style.display = 'none';
    
    renderSessions();
    messageInput.focus();
}

function renderSessions() {
    if (!sessionsList) return;
    sessionsList.innerHTML = sessions.map((s, i) => `
        <div class="session-item ${s.id === currentSession ? 'active' : ''}" onclick="switchSession('${s.id}')">
            <div class="session-name">${s.name}</div>
            <div class="session-model">${s.model}</div>
        </div>
    `).join('');
}

function switchSession(sessionId) {
    currentSession = sessionId;
    localStorage.setItem('chat_session_id', sessionId);
    
    // Clear and reload
    messagesContainer.innerHTML = '<div class="typing" id="typing"><span></span><span></span><span></span></div>';
    loadHistory();
    renderSessions();
}

// Stats
function toggleStats() {
    statsPanel.classList.toggle('show');
    if (statsPanel.classList.contains('show')) {
        loadStats();
    }
}

// Snackbar
function showSnackbar(source, text) {
    const icons = {
        'wikipedia': '🌐',
        'books': '📚',
        'local': '💬',
        'thinking': '🧠'
    };
    
    snackbar.querySelector('.snackbar-icon').textContent = icons[source] || '💬';
    snackbar.querySelector('.snackbar-text').textContent = text || 'Thinking...';
    snackbar.classList.add('show');
    
    setTimeout(() => {
        snackbar.classList.remove('show');
    }, 2000);
}

// Load history
async function loadHistory() {
    try {
        console.log('Loading history for session:', currentSession);
        const response = await fetch('/history?session_id=' + currentSession);
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            console.log('Loaded', data.history.length, 'messages');
            welcomeScreen.style.display = 'none';
            messagesContainer.style.display = 'block';
            
            data.history.forEach(msg => {
                addMessageToChat(msg.user, 'user');
                addMessageToChat(msg.bot, 'bot', msg.time, msg.source);
            });
        } else {
            console.log('No history, showing welcome');
            welcomeScreen.style.display = 'flex';
            messagesContainer.style.display = 'none';
        }
    } catch (error) {
        console.log('Failed to load history:', error);
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
        if (statsContent) {
            statsContent.innerHTML = '<div class="loading">Failed to load stats</div>';
        }
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
                    <div class="stat-value">${data.stats.total_queries}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Avg Response</div>
                    <div class="stat-value">${avgTime.toFixed(2)}s</div>
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
                    <span class="nn-status">loaded</span>
                </div>
                <div class="nn-details">
                    <span>${nn.architecture}</span>
                    <span>${nn.weights} weights</span>
                    ${nn.training_count !== undefined ? '<span>Trained: ' + nn.training_count + 'x</span>' : ''}
                </div>
                ${nn.training_count !== undefined ? '<div class="progress-bar"><div class="progress-fill neural" style="width: ' + nnProgress + '%"></div></div>' : ''}
            </div>
        `;
    }
    
    html += '</div>';
    if (statsContent) {
        statsContent.innerHTML = html;
    }
}

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) return hours + 'h ' + minutes + 'm';
    if (minutes > 0) return minutes + 'm ' + secs + 's';
    return secs + 's';
}

// Messages
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) {
        console.log('Empty message, ignoring');
        return;
    }

    console.log('Sending message:', message);

    // Switch to message view
    welcomeScreen.style.display = 'none';
    messagesContainer.style.display = 'block';
    
    // Add user message
    addMessageToChat(message, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Show typing
    const typingEl = document.getElementById('typing');
    if (typingEl) typingEl.classList.add('show');
    
    showSnackbar('thinking', 'Processing...');
    chat.scrollTop = chat.scrollHeight;

    try {
        console.log('Calling /chat endpoint...');
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, session_id: currentSession })
        });

        console.log('Response received:', response.status);
        const data = await response.json();
        console.log('Data:', data);
        
        // Hide typing
        if (typingEl) typingEl.classList.remove('show');
        
        // Show source
        showSnackbar(data.source, 'From ' + (data.source === 'wikipedia' ? 'Wikipedia' : data.source === 'books' ? 'Books' : 'Local'));
        
        // Add bot message
        addMessageToChat(data.response, 'bot', data.response_time, data.source);
        
        // Update stats
        if (statsPanel.classList.contains('show')) {
            loadStats();
        }
    } catch (error) {
        console.error('Error:', error);
        if (typingEl) typingEl.classList.remove('show');
        showSnackbar('local', 'Error occurred');
        addMessageToChat('Sorry, something went wrong. Please try again.', 'bot');
    }

    sendBtn.disabled = false;
    messageInput.focus();
}

function addMessageToChat(text, type, responseTime, source) {
    const div = document.createElement('div');
    div.className = 'message ' + type;
    
    var content = '';
    
    if (type === 'bot' && text.includes('— Source:')) {
        var parts = text.split('— Source:');
        content = '<div>' + parts[0].trim() + '</div><div class="source">— Source:' + parts[1] + '</div>';
    } else {
        content = text;
    }
    
    if (type === 'bot' && responseTime != null) {
        content += '<div class="response-time">' + responseTime.toFixed(2) + 's</div>';
    }
    
    if (type === 'bot' && source) {
        var badge = source === 'wikipedia' ? '🌐 Wikipedia' : source === 'books' ? '📚 Books' : '💬 Local';
        content += '<div class="source-badge">' + badge + '</div>';
    }
    
    div.innerHTML = content;
    messagesContainer.insertBefore(div, typing);
    chat.scrollTop = chat.scrollHeight;
}
