let currentSessionId = "sess-" + Math.random().toString(36).substring(2, 9);
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const badge = document.getElementById('cache-badge');


const pdfList = document.getElementById('pdf-list');

// Local cache of uploaded PDFs registry
let pdfRegistry = {};

// Chat UI functions
function addMessage(text, isUser = false, cacheHit = false, citations = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    let citationHtml = '';
    if (citations && citations.length > 0) {
        citationHtml = '<div class="citation-container">' + citations.map(c => {
            const url = c.pdf_url || '';
            const dispName = c.source.replace(/_/g, ' ');
            return `<span class="source-tag" data-url="${url}" data-source="${c.source}" data-page="${c.page_number}">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                ${dispName} (Page ${c.page_number})
            </span>`;
        }).join('') + '</div>';
    }

    msgDiv.innerHTML = `
        <div class="avatar">${isUser ? 'U' : 'AI'}</div>
        <div class="bubble">
            <div>${text}</div>
            ${citationHtml}
        </div>
    `;
    
    // Attach click listeners to citation tags in this message
    msgDiv.querySelectorAll('.source-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const url = tag.getAttribute('data-url');
            const source = tag.getAttribute('data-source');
            const page = tag.getAttribute('data-page');
            openPdfViewer(url, source, page);
        });
    });
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    if (!isUser) {
        if (cacheHit) {
            badge.classList.remove('hidden');
            setTimeout(() => badge.classList.add('hidden'), 3000);
        }
    }
    return msgDiv;
}

function showTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant typing-msg';
    msgDiv.innerHTML = `
        <div class="avatar">AI</div>
        <div class="bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
}

// PDF Viewer Actions
function openPdfViewer(url, source, page = 1) {
    const viewerPanel = document.getElementById('pdf-viewer-panel');
    const iframe = document.getElementById('pdf-iframe');
    const placeholder = document.getElementById('pdf-placeholder');
    const title = document.getElementById('pdf-viewer-title');
    
    // Resolve URL from local registry cache if not directly provided
    const finalUrl = url || (pdfRegistry[source] ? pdfRegistry[source].cloudinary_url : '');
    
    if (!finalUrl) {
        alert(`The document "${source}" does not have a Cloudinary PDF link in the registry.`);
        return;
    }
    
    // Slide open the panel
    viewerPanel.classList.remove('closed');
    
    // Show iframe, hide placeholder
    placeholder.classList.add('hidden');
    iframe.classList.remove('hidden');
    
    // Set title
    const displayName = source.replace(/_/g, ' ');
    title.textContent = `${displayName} (Page ${page})`;
    
    // Set iframe source with page hash (browser native PDF viewer support)
    iframe.src = `${finalUrl}#page=${page}`;
}

const closeViewerBtn = document.getElementById('close-viewer-btn');
closeViewerBtn.addEventListener('click', () => {
    const viewerPanel = document.getElementById('pdf-viewer-panel');
    const iframe = document.getElementById('pdf-iframe');
    const placeholder = document.getElementById('pdf-placeholder');
    
    viewerPanel.classList.add('closed');
    iframe.src = '';
    iframe.classList.add('hidden');
    placeholder.classList.remove('hidden');
    
    // Clear sidebar active highlights
    if (pdfList) {
        pdfList.querySelectorAll('.pdf-item').forEach(i => i.classList.remove('active'));
    }
});

// Fetch and Render PDF list
async function fetchAndRenderPdfs() {
    try {
        const res = await fetch('/admin/pdfs');
        if (!res.ok) throw new Error("Failed to fetch PDFs");
        const data = await res.json();
        pdfRegistry = data.pdfs || {};
        
        const keys = Object.keys(pdfRegistry);
        
        if (pdfList) {
            if (keys.length === 0) {
                pdfList.innerHTML = '<div class="no-pdfs-msg">No PDFs uploaded yet.</div>';
                return;
            }
            
            pdfList.innerHTML = keys.map(key => {
                const item = pdfRegistry[key];
                const displayName = key.replace(/_/g, ' ');
                return `
                    <div class="pdf-item" data-source="${key}" data-url="${item.cloudinary_url}">
                        <div class="pdf-item-icon">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                            </svg>
                        </div>
                        <div class="pdf-item-info">
                            <div class="pdf-item-name" title="${item.filename}">${displayName}</div>
                            <div class="pdf-item-meta">PDF Document</div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Attach click listeners to list items
            pdfList.querySelectorAll('.pdf-item').forEach(item => {
                item.addEventListener('click', () => {
                    const source = item.getAttribute('data-source');
                    const url = item.getAttribute('data-url');
                    
                    // Highlight active item
                    pdfList.querySelectorAll('.pdf-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                    
                    openPdfViewer(url, source, 1);
                });
            });
        }
    } catch (err) {
        console.error("Error fetching PDF list:", err);
        if (pdfList) {
            pdfList.innerHTML = '<div class="no-pdfs-msg" style="color: #ef4444;">Error loading PDFs list.</div>';
        }
    }
}

// --- Auth State ---
let authToken = localStorage.getItem('shariagpt_token');

async function updateAuthUI() {
    const unauthControls = document.getElementById('unauth-controls');
    const authControls = document.getElementById('auth-controls');
    const userNameEl = document.getElementById('auth-user-name');
    const userTypeEl = document.getElementById('auth-user-type');
    const landingPage = document.getElementById('landing-page');
    const appContent = document.getElementById('app-content');
    const chatHistory = document.getElementById('chat-history-section');
    
    if (!authToken) {
        unauthControls.classList.remove('hidden');
        authControls.classList.add('hidden');
        landingPage.classList.remove('hidden');
        appContent.classList.add('hidden');
        if (chatHistory) chatHistory.classList.add('hidden');
        return;
    }
    
    try {
        const res = await fetch('/auth/me', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (res.ok) {
            const user = await res.json();
            unauthControls.classList.add('hidden');
            authControls.classList.remove('hidden');
            landingPage.classList.add('hidden');
            appContent.classList.remove('hidden');
            if (chatHistory) chatHistory.classList.remove('hidden');
            
            userNameEl.textContent = user.name;
            userTypeEl.textContent = user.account_type;
            
            // Fetch chat history for logged in user
            fetchUserChats();
        } else {
            // Token invalid or expired
            logout();
        }
    } catch (err) {
        console.error("Auth check failed", err);
    }
}

function logout() {
    authToken = null;
    localStorage.removeItem('shariagpt_token');
    updateAuthUI();
}

document.getElementById('btn-logout').addEventListener('click', logout);

document.getElementById('btn-export-data').addEventListener('click', async () => {
    if (!authToken) return;
    try {
        const res = await fetch('/auth/export', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (!res.ok) throw new Error("Failed to export data");
        const data = await res.json();
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `shariagpt_export_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        alert("Failed to export data: " + e.message);
    }
});

document.getElementById('btn-delete-account').addEventListener('click', async () => {
    if (!authToken) return;
    if (!confirm("Are you ABSOLUTELY sure you want to delete your account?\n\nThis action cannot be undone and will permanently erase all your chats and profile data.")) {
        return;
    }
    
    try {
        const res = await fetch('/auth/account', {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to delete account");
        }
        alert("Your account has been permanently deleted.");
        logout();
    } catch (e) {
        alert("Failed to delete account: " + e.message);
    }
});

// --- Chat API Call ---
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    const userMsgDiv = addMessage(message, true);
    chatInput.value = '';
    
    // Show typing
    const typingMsg = showTypingIndicator();

    try {
        const payload = {
            session_id: currentSessionId,
            message: message
        };

        const headers = { 'Content-Type': 'application/json' };
        if (authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }

        const res = await fetch('/chat', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        
        // Remove typing indicator
        typingMsg.remove();
        
        // Check if PII was redacted
        if (data.pii_detected && data.pii_detected.length > 0) {
            const piiBadge = document.createElement('div');
            piiBadge.className = 'pii-badge';
            piiBadge.innerHTML = `🛡️ PII Masked: ${data.pii_detected.join(', ')}`;
            userMsgDiv.querySelector('.bubble').appendChild(piiBadge);
        }

        addMessage(data.response, false, data.cache_hit, data.citations || []);

        // Refresh sidebar to capture the new chat title if this was a new thread
        fetchUserChats();

    } catch (err) {
        typingMsg.remove();
        addMessage("Sorry, I encountered an error connecting to the server.", false);
    }
});



// Initial fetch on page load
fetchAndRenderPdfs();
updateAuthUI();

// --- Auth Modal Logic ---
const authModal = document.getElementById('auth-modal');
const btnLoginModal = document.getElementById('btn-login-modal');
const btnRegisterModal = document.getElementById('btn-register-modal');
const btnLpLogin = document.getElementById('btn-lp-login');
const btnLpRegister = document.getElementById('btn-lp-register');
const btnCloseModal = document.getElementById('btn-close-modal');
const authForm = document.getElementById('auth-form');
const registerFields = document.getElementById('register-fields');
const modalTitle = document.getElementById('modal-title');
const authError = document.getElementById('auth-error');

let isRegistering = false;
let pendingToken = null;

function openLoginModal() {
    isRegistering = false;
    modalTitle.textContent = 'Login';
    registerFields.classList.add('hidden');
    authError.classList.add('hidden');
    authModal.classList.remove('hidden');
}

function openRegisterModal() {
    isRegistering = true;
    modalTitle.textContent = 'Register';
    registerFields.classList.remove('hidden');
    authError.classList.add('hidden');
    document.querySelectorAll('#register-fields input').forEach(el => el.required = true);
    authModal.classList.remove('hidden');
}

btnLoginModal.addEventListener('click', openLoginModal);
btnLpLogin.addEventListener('click', openLoginModal);

btnRegisterModal.addEventListener('click', openRegisterModal);
btnLpRegister.addEventListener('click', openRegisterModal);

btnCloseModal.addEventListener('click', () => {
    authModal.classList.add('hidden');
    authForm.reset();
});

authForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    authError.classList.add('hidden');
    
    const email = document.getElementById('auth-email').value.trim();
    const password = document.getElementById('auth-password').value;
    
    if (isRegistering) {
        const name = document.getElementById('auth-name').value.trim();
        const emiratesId = document.getElementById('auth-emirates-id').value.trim();
        const accNumber = document.getElementById('auth-account-number').value.trim();
        const accType = document.getElementById('auth-account-type').value;
        const balance = document.getElementById('auth-balance').value.trim();
        
        try {
            const res = await fetch('/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email, password, name, emirates_id: emiratesId,
                    account_number: accNumber, account_type: accType, balance
                })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);
            
            // Show QR Code modal
            authModal.classList.add('hidden');
            show2FAModal(data.qr_code_base64);
            
            // Proceed to login them immediately to get pending token
            const loginRes = await fetch('/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const loginData = await loginRes.json();
            pendingToken = loginData.token;
            
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        }
    } else {
        // Login
        try {
            const res = await fetch('/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);
            
            authModal.classList.add('hidden');
            authForm.reset();
            
            if (data.requires_2fa) {
                pendingToken = data.token;
                show2FAModal();
            }
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        }
    }
});

// --- 2FA Modal Logic ---
const twofaModal = document.getElementById('twofa-modal');
const qrContainer = document.getElementById('qr-container');
const qrImage = document.getElementById('qr-image');
const btnVerify2fa = document.getElementById('btn-verify-2fa');
const totpInput = document.getElementById('totp-code');
const twofaError = document.getElementById('twofa-error');

function show2FAModal(qrBase64 = null) {
    if (qrBase64) {
        qrImage.src = `data:image/png;base64,${qrBase64}`;
        qrContainer.classList.remove('hidden');
    } else {
        qrContainer.classList.add('hidden');
    }
    totpInput.value = '';
    twofaError.classList.add('hidden');
    twofaModal.classList.remove('hidden');
}

btnVerify2fa.addEventListener('click', async () => {
    const code = totpInput.value.trim();
    if (!code || code.length !== 6) return;
    
    try {
        const res = await fetch('/auth/verify-2fa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                token: pendingToken,
                code: code
            })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        
        // Success! Save real token
        authToken = data.token;
        localStorage.setItem('shariagpt_token', authToken);
        
        twofaModal.classList.add('hidden');
        updateAuthUI();
        
    } catch (err) {
        twofaError.textContent = err.message || "Invalid code";
        twofaError.classList.remove('hidden');
    }
});

// --- Chat History Logic ---
async function fetchUserChats() {
    if (!authToken) return;
    
    try {
        const res = await fetch('/chat/sessions', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) {
            const data = await res.json();
            renderChatHistory(data.chats);
        }
    } catch (e) {
        console.error("Failed to fetch user chats", e);
    }
}

function renderChatHistory(chats) {
    const list = document.getElementById('chat-history-list');
    if (!chats || chats.length === 0) {
        list.innerHTML = '<div class="no-chats-msg">No recent chats.</div>';
        return;
    }
    
    list.innerHTML = chats.map(chat => `
        <div class="chat-item ${chat.session_id === currentSessionId ? 'active' : ''}" data-id="${chat.session_id}">
            💬 ${chat.title || 'New Chat'}
        </div>
    `).join('');
    
    list.querySelectorAll('.chat-item').forEach(item => {
        item.addEventListener('click', () => {
            loadChatHistory(item.getAttribute('data-id'));
        });
    });
}

async function loadChatHistory(id) {
    if (!authToken) return;
    currentSessionId = id;
    
    // Highlight active
    document.querySelectorAll('.chat-item').forEach(i => {
        i.classList.toggle('active', i.getAttribute('data-id') === id);
    });
    
    // Clear current window except the intro message
    const msgs = chatMessages.querySelectorAll('.message');
    msgs.forEach((m, i) => {
        if (i > 0) m.remove(); // Keep first welcome message
    });
    
    try {
        const res = await fetch(`/chat/sessions/${id}`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) {
            const data = await res.json();
            const history = data.history || [];
            
            history.forEach(turn => {
                if (turn.role === 'user') {
                    addMessage(turn.content, true);
                } else if (turn.role === 'assistant') {
                    addMessage(turn.content, false);
                }
            });
        }
    } catch (e) {
        console.error("Failed to load chat", e);
    }
}

document.getElementById('btn-new-chat').addEventListener('click', () => {
    currentSessionId = "sess-" + Math.random().toString(36).substring(2, 9);
    
    // Highlight none
    document.querySelectorAll('.chat-item').forEach(i => i.classList.remove('active'));
    
    // Clear chat window
    const msgs = chatMessages.querySelectorAll('.message');
    msgs.forEach((m, i) => {
        if (i > 0) m.remove();
    });
});
