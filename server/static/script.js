const chat = document.getElementById('chat');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');

// Send on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Focus input on load
messageInput.focus();

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
        
        // Add bot response
        addMessage(data.response, 'bot');
    } catch (error) {
        typing.classList.remove('show');
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
    }

    sendBtn.disabled = false;
    messageInput.focus();
}

function addMessage(text, type) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    
    // Parse source attribution
    if (type === 'bot' && text.includes('— Source:')) {
        const parts = text.split('— Source:');
        div.innerHTML = `
            <div>${parts[0].trim()}</div>
            <div class="source">— Source:${parts[1]}</div>
        `;
    } else {
        div.textContent = text;
    }
    
    // Insert before typing indicator
    chat.insertBefore(div, typing);
    chat.scrollTop = chat.scrollHeight;
}
