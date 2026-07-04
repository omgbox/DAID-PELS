// State
let sessionId = localStorage.getItem('sid') || ('s' + Date.now());
localStorage.setItem('sid', sessionId);
let sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
let statsOpen = false;

// Elements
const $ = id => document.getElementById(id);
const input = $('input');
const messages = $('messages');
const welcome = $('welcome');
const typing = $('typing');
const snackbar = $('snackbar');
const statsPanel = $('statsPanel');
const sessionsEl = $('sessions');
const sendBtn = $('sendBtn');
const topbarTitle = $('topbarTitle');

// Init
console.log('Script loaded');
console.log('Elements:', { input, messages, welcome, typing, sendBtn });

input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
});
input.focus();

loadSessions();
loadHistory();

console.log('Initialization complete');

// Send message
async function send() {
    console.log('send() called');
    const text = input.value.trim();
    console.log('Input text:', text);
    if (!text) {
        console.log('No text, returning');
        return;
    }

    console.log('Hiding welcome, adding user msg');
    welcome.classList.add('hidden');
    addMsg(text, 'user');
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    typing.classList.add('show');
    snackbarShow('Processing...');
    messages.scrollTop = messages.scrollHeight;

    try {
        console.log('Fetching /chat...');
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });
        console.log('Response status:', res.status);
        const data = await res.json();
        console.log('Response data:', data);
        typing.classList.remove('show');
        
        const sourceText = data.source === 'wikipedia' ? 'Wikipedia' : data.source === 'books' ? 'Books' : 'Local';
        snackbarShow(sourceText);
        
        console.log('Adding bot message');
        addMsg(data.response, 'bot', data.response_time, data.source);
        
    } catch (e) {
        console.error('Error:', e);
        typing.classList.remove('show');
        addMsg('Error: ' + e.message, 'bot');
    }

    sendBtn.disabled = false;
    input.focus();
}

// Add message to chat
function addMsg(text, type, time, source) {
    console.log('addMsg called:', type, text.substring(0, 50));
    const div = document.createElement('div');
    div.className = 'msg msg-' + type;
    
    let html = text;
    if (type === 'bot') {
        if (text.includes('— Source:')) {
            const parts = text.split('— Source:');
            html = parts[0].trim() + '<div class="msg-source">— Source:' + parts[1] + '</div>';
        }
        if (time) html += '<div class="msg-time">' + time.toFixed(1) + 's</div>';
        if (source) {
            const badge = source === 'wikipedia' ? 'Wikipedia' : source === 'books' ? 'Books' : 'Local';
            html += '<div class="msg-badge">' + badge + '</div>';
        }
    }
    
    div.innerHTML = html;
    console.log('Inserting before typing element');
    messages.insertBefore(div, typing);
    messages.scrollTop = messages.scrollHeight;
    console.log('Message added to DOM');
}

// Snackbar
function snackbarShow(text) {
    snackbar.textContent = text;
    snackbar.classList.add('show');
    setTimeout(() => snackbar.classList.remove('show'), 2000);
}

// Sessions
function loadSessions() {
    sessionsEl.innerHTML = sessions.map(s =>
        '<div class="session' + (s.id === sessionId ? ' active' : '') + '" onclick="switchSession(\'' + s.id + '\')">' +
        '<div class="session-title">' + s.name + '</div>' +
        '<div class="session-sub">' + s.model + '</div></div>'
    ).join('');
}

function switchSession(id) {
    sessionId = id;
    localStorage.setItem('sid', id);
    messages.innerHTML = '<div class="welcome" id="welcome"><h1>DAID-PELS</h1><p>Ask me anything!</p></div><div class="typing" id="typing"><span></span><span></span><span></span></div>';
    loadHistory();
    loadSessions();
}

function newSession(name) {
    sessionId = 's' + Date.now();
    localStorage.setItem('sid', sessionId);
    sessions.unshift({ id: sessionId, name: name || 'New Chat', model: 'Wikipedia + Local' });
    if (sessions.length > 20) sessions = sessions.slice(0, 20);
    localStorage.setItem('sessions', JSON.stringify(sessions));
    
    messages.innerHTML = '<div class="welcome" id="welcome"><h1>DAID-PELS</h1><p>Ask me anything!</p></div><div class="typing" id="typing"><span></span><span></span><span></span></div>';
    loadSessions();
    input.focus();
}

// Load history
async function loadHistory() {
    try {
        const res = await fetch('/history?session_id=' + sessionId);
        const data = await res.json();
        
        const w = document.getElementById('welcome');
        if (data.history && data.history.length > 0) {
            if (w) w.classList.add('hidden');
            data.history.forEach(m => {
                addMsg(m.user, 'user');
                addMsg(m.bot, 'bot', m.time, m.source);
            });
        }
    } catch (e) {}
}

// Stats
function toggleStats() {
    statsOpen = !statsOpen;
    if (statsOpen) {
        loadStats();
        statsPanel.classList.add('show');
    } else {
        statsPanel.classList.remove('show');
    }
}

async function loadStats() {
    try {
        const res = await fetch('/stats');
        const data = await res.json();
        renderStats(data);
    } catch (e) {}
}

function renderStats(d) {
    const nn = Object.values(d.neural_networks || {});
    statsPanel.innerHTML = '<div class="stats-grid">' +
        '<div class="stat"><div class="stat-label">Uptime</div><div class="stat-val">' + formatTime(d.uptime) + '</div></div>' +
        '<div class="stat"><div class="stat-label">Memory</div><div class="stat-val">' + d.memory_mb + ' MB</div></div>' +
        '<div class="stat"><div class="stat-label">Queries</div><div class="stat-val">' + d.stats.total_queries + '</div></div>' +
        '<div class="stat"><div class="stat-label">Avg Time</div><div class="stat-val">' + d.stats.avg_response_time.toFixed(1) + 's</div></div>' +
        '</div>' +
        '<div style="padding:0 20px 16px;display:flex;gap:8px;flex-wrap:wrap">' +
        nn.map(n => '<div style="background:var(--bg);padding:8px 12px;border-radius:8px;font-size:0.8em"><b>' + n.name + '</b> ' + n.architecture + '</div>').join('') +
        '</div>';
}

function formatTime(s) {
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
    return h > 0 ? h+'h '+m+'m' : m > 0 ? m+'m' : Math.floor(s)+'s';
}

// Sidebar toggle
function toggleSidebar() {
    $('sidebar').classList.toggle('show');
}
