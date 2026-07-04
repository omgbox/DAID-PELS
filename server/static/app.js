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
    const text = input.value.trim();
    if (!text) return;

    // Hide welcome, show typing
    const w = document.getElementById('welcome');
    const t = document.getElementById('typing');
    if (w) w.classList.add('hidden');
    if (t) t.classList.add('show');
    
    addMsg(text, 'user');
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    snackbarShow('Processing...');
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });
        const data = await res.json();
        
        if (t) t.classList.remove('show');
        
        const sourceText = data.source === 'wikipedia' ? 'Wikipedia' : data.source === 'books' ? 'Books' : 'Local';
        snackbarShow(sourceText);
        
        addMsg(data.response, 'bot', data.response_time, data.source);
        
    } catch (e) {
        const t2 = document.getElementById('typing');
        if (t2) t2.classList.remove('show');
        addMsg('Error: ' + e.message, 'bot');
    }

    sendBtn.disabled = false;
    input.focus();
}

// Add message to chat
function addMsg(text, type, time, source) {
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
    // Always get fresh reference to typing element
    const t = document.getElementById('typing');
    if (t) {
        messages.insertBefore(div, t);
    } else {
        messages.appendChild(div);
    }
    messages.scrollTop = messages.scrollHeight;
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
    // Clear messages but keep structure
    const msgList = messages.querySelectorAll('.msg');
    msgList.forEach(m => m.remove());
    // Show welcome
    const w = document.getElementById('welcome');
    if (w) w.classList.remove('hidden');
    loadHistory();
    loadSessions();
}

function newSession(name) {
    sessionId = 's' + Date.now();
    localStorage.setItem('sid', sessionId);
    sessions.unshift({ id: sessionId, name: name || 'New Chat', model: 'Wikipedia + Local' });
    if (sessions.length > 20) sessions = sessions.slice(0, 20);
    localStorage.setItem('sessions', JSON.stringify(sessions));
    
    // Clear messages but keep structure
    const msgList = messages.querySelectorAll('.msg');
    msgList.forEach(m => m.remove());
    const w = document.getElementById('welcome');
    if (w) w.classList.remove('hidden');
    
    loadSessions();
    input.focus();
}

// Load history
async function loadHistory() {
    try {
        console.log('Loading history for session:', sessionId);
        const res = await fetch('/history?session_id=' + sessionId);
        const data = await res.json();
        console.log('History loaded:', data.history ? data.history.length : 0, 'messages');
        
        const w = document.getElementById('welcome');
        const t = document.getElementById('typing');
        
        if (data.history && data.history.length > 0) {
            if (w) w.classList.add('hidden');
            data.history.forEach(m => {
                addMsg(m.user, 'user');
                addMsg(m.bot, 'bot', m.time, m.source);
            });
        } else {
            if (w) w.classList.remove('hidden');
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
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
