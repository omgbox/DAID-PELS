const chat = document.getElementById('chat');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');
const statsPanel = document.getElementById('statsPanel');
const statsContent = document.getElementById('statsContent');

// Send on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Focus input on load
messageInput.focus();

// Load stats on page load
loadStats();

function toggleStats() {
    statsPanel.classList.toggle('show');
    if (statsPanel.classList.contains('show')) {
        loadStats();
    }
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        renderStats(data);
    } catch (error) {
        statsContent.innerHTML = '<div class="loading">Failed to load stats</div>';
    }
}

function renderStats(data) {
    const uptime = formatUptime(data.uptime);
    
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
                    <div class="stat-value">${data.stats.avg_response_time.toFixed(2)}s</div>
                </div>
            </div>
        </div>
        
        <div class="stat-section">
            <h4>Neural Networks</h4>
    `;
    
    for (const [key, nn] of Object.entries(data.neural_networks)) {
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

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';
    sendBtn.disabled = true;

    // Show typing indicator
    typing.classList.add('show');
    chat.scrollTop = chat.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        
        // Hide typing
        typing.classList.remove('show');
        
        // Add bot response with response time
        addMessage(data.response, 'bot', data.response_time);
        
        // Update stats
        if (statsPanel.classList.contains('show')) {
            loadStats();
        }
    } catch (error) {
        typing.classList.remove('show');
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
    }

    sendBtn.disabled = false;
    messageInput.focus();
}

function addMessage(text, type, responseTime = null) {
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
    
    div.innerHTML = content;
    
    // Insert before typing indicator
    chat.insertBefore(div, typing);
    chat.scrollTop = chat.scrollHeight;
}
