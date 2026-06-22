// Gapura RAG — Chat Application Logic
// Handles SSE answer delivery, file upload, and UI interactions.

const DOM = {
    chatContainer: () => document.getElementById('chat-container'),
    messages: () => document.getElementById('messages'),
    welcome: () => document.getElementById('chat-welcome'),
    chatForm: () => document.getElementById('chat-form'),
    chatInput: () => document.getElementById('chat-input'),
    btnSend: () => document.getElementById('btn-send'),
    btnClear: () => document.getElementById('btn-clear-chat'),
    langSelect: () => document.getElementById('lang-select'),
    statsMini: () => document.getElementById('stats-mini'),
    dropZone: () => document.getElementById('drop-zone'),
    fileInput: () => document.getElementById('file-input'),
    uploadLog: () => document.getElementById('upload-log'),
};

const chatHistory = [];
const VA_TOKEN_STORAGE_KEY = 'gapura_va_token';

function initVirtualAssistantToken() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('va_token');
    if (!token) return;

    sessionStorage.setItem(VA_TOKEN_STORAGE_KEY, token);
    params.delete('va_token');

    const nextSearch = params.toString();
    const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
    window.history.replaceState({}, document.title, nextUrl);
}

function getVirtualAssistantToken() {
    return sessionStorage.getItem(VA_TOKEN_STORAGE_KEY) || '';
}

function getUiStrings(language) {
    if (language === 'id') {
        return {
            evidenceTitle: 'Bukti',
            grounded: 'Berdasarkan dokumen',
            partial: 'Sebagian didukung dokumen',
            unsupported: 'Tidak didukung dokumen',
            supplementNote: 'Jawaban mencakup tambahan yang diberi label di luar dokumen.',
            pageLabel: 'Halaman',
            noResponse: 'Tidak ada respons diterima.',
            sourcesTitle: 'Sumber',
        };
    }

    return {
        evidenceTitle: 'Evidence',
        grounded: 'Document-grounded',
        partial: 'Partially grounded',
        unsupported: 'Not supported by docs',
        supplementNote: 'Answer includes a labeled outside-document supplement.',
        pageLabel: 'Page',
        noResponse: 'No response received.',
        sourcesTitle: 'Sources',
    };
}

function closeSidebar() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebar-overlay').classList.remove('open');
}

function openSidebar() {
    document.getElementById('sidebar').classList.add('open');
    document.getElementById('sidebar-overlay').classList.add('open');
}

function switchPanel(panelName) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.bottom-nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

    document.querySelectorAll(`[data-panel="${panelName}"]`).forEach(b => b.classList.add('active'));
    const panel = document.getElementById(`panel-${panelName}`);
    if (panel) panel.classList.add('active');
}

function initNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchPanel(btn.dataset.panel);
            closeSidebar();
        });
    });

    document.querySelectorAll('.bottom-nav-btn[data-panel]').forEach(btn => {
        btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
    });

    document.getElementById('btn-menu')?.addEventListener('click', openSidebar);
    document.getElementById('btn-close-sidebar')?.addEventListener('click', closeSidebar);
    document.getElementById('sidebar-overlay')?.addEventListener('click', closeSidebar);
    document.getElementById('btn-bottom-settings')?.addEventListener('click', openSidebar);
}

function initChatInput() {
    const input = DOM.chatInput();
    const btn = DOM.btnSend();

    input.addEventListener('input', () => {
        btn.disabled = !input.value.trim();
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (input.value.trim()) DOM.chatForm().requestSubmit();
        }
    });

    DOM.chatForm().addEventListener('submit', (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question) return;

        input.value = '';
        input.style.height = 'auto';
        btn.disabled = true;
        sendMessage(question);
    });

    DOM.btnClear().addEventListener('click', clearChat);
}

function initUploadPanel() {
    const dropZone = DOM.dropZone();
    const fileInput = DOM.fileInput();
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            Array.from(fileInput.files).forEach(uploadFile);
            fileInput.value = '';
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
        if (!files.length) {
            addUploadLog('⚠️', 'No PDF files found', 'Please drop PDF files only.', 'error');
            return;
        }
        files.forEach(uploadFile);
    });

    document.getElementById('btn-menu-upload')?.addEventListener('click', openSidebar);
}

async function uploadFile(file) {
    const item = addUploadLog('⏳', file.name, 'Uploading...', '');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();

        if (!res.ok) {
            updateUploadLog(item, '❌', file.name, data.error || `Upload failed (${res.status})`, 'error');
            return;
        }

        if (data.skipped) {
            updateUploadLog(item, '⏭️', file.name, 'Already exists in knowledge base (skipped)', 'success');
            return;
        }

        const pages = data.pages === -1 ? 'Processing...' : `${data.pages} pages`;
        const chunks = data.chunks === -1 ? '' : `, ${data.chunks} chunks`;
        updateUploadLog(item, '✅', file.name, `${pages}${chunks}`, 'success');
        loadStats();
    } catch (err) {
        updateUploadLog(item, '❌', file.name, `Connection error: ${err.message}`, 'error');
    }
}

function addUploadLog(statusIcon, filename, detail, className) {
    const log = DOM.uploadLog();
    if (!log) return null;

    const item = document.createElement('div');
    item.className = 'upload-item' + (className ? ` ${className}` : '');
    item.innerHTML = `
        <span class="upload-status">${statusIcon}</span>
        <div class="upload-info">
            <div class="upload-filename">${escapeHtml(filename)}</div>
            <div class="upload-detail">${escapeHtml(detail)}</div>
        </div>
    `;
    log.prepend(item);
    return item;
}

function updateUploadLog(item, statusIcon, filename, detail, className) {
    if (!item) return;
    item.className = 'upload-item' + (className ? ` ${className}` : '');
    item.querySelector('.upload-status').textContent = statusIcon;
    item.querySelector('.upload-filename').textContent = filename;
    item.querySelector('.upload-detail').textContent = detail;
}

function hideWelcome() {
    const welcome = DOM.welcome();
    if (welcome) welcome.classList.add('hidden');
}

function clearChat() {
    DOM.messages().innerHTML = '';
    chatHistory.length = 0;
    const welcome = DOM.welcome();
    if (welcome) welcome.classList.remove('hidden');
}

function escapeHtml(text) {
    const safeText = String(text ?? '');
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return safeText.replace(/[&<>"']/g, c => map[c]);
}

function formatMessageHtml(text) {
    const paragraphs = String(text ?? '').trim().split(/\n{2,}/).filter(Boolean);
    if (!paragraphs.length) return '';
    return paragraphs
        .map(paragraph => `<p>${escapeHtml(paragraph).replace(/\n/g, '<br>')}</p>`)
        .join('');
}

function tryParseAssistantPayload(text) {
    const trimmed = String(text ?? '').trim();
    if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) {
        return null;
    }

    try {
        const payload = JSON.parse(trimmed);
        return typeof payload?.answer === 'string' ? payload : null;
    } catch {
        return null;
    }
}

function normalizeAssistantText(text, language = 'en') {
    const payload = tryParseAssistantPayload(text);
    if (!payload) {
        return { text: String(text ?? ''), supplementUsed: false };
    }

    let answerText = String(payload.answer ?? '').trim();
    const supplement = String(payload.supplement ?? '').trim();
    if (supplement) {
        const label = language === 'id' ? 'Di luar dokumen:' : 'Outside the documents:';
        answerText = `${answerText}\n\n${label}\n${supplement}`;
    }

    return {
        text: answerText,
        supplementUsed: Boolean(supplement),
        groundingStatus: payload.grounding_status || null,
    };
}

function buildGroundingMetaHtml(data) {
    const strings = getUiStrings(data.language);
    const labelMap = {
        grounded: strings.grounded,
        partial: strings.partial,
        unsupported: strings.unsupported,
    };
    const label = labelMap[data.grounding_status] || strings.unsupported;
    const note = data.supplement_used
        ? `<div class="msg-grounding-note">${escapeHtml(strings.supplementNote)}</div>`
        : '';

    return `
        <div class="msg-grounding">
            <span class="grounding-pill ${escapeHtml(data.grounding_status || 'unsupported')}">${escapeHtml(label)}</span>
        </div>
        ${note}
    `;
}

function buildEvidenceHtml(data) {
    if (!data.evidence?.length) return '';
    const strings = getUiStrings(data.language);

    return `
        <div class="msg-citations">
            <strong>${escapeHtml(strings.evidenceTitle)}</strong>
            ${data.evidence.map(item => `
                <div class="msg-citation-item">
                    <div class="msg-citation-header">
                        <span class="msg-citation-id">${escapeHtml(item.id || '')}</span>
                        <span class="msg-citation-source">${escapeHtml(item.source)}, ${escapeHtml(strings.pageLabel)} ${escapeHtml(item.page)}</span>
                    </div>
                    <div class="msg-citation-snippet">${escapeHtml(item.snippet || '')}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function appendMessage(role, content) {
    hideWelcome();
    const container = DOM.messages();
    const avatarLabel = role === 'user' ? 'You' : 'AI';

    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;
    msgEl.innerHTML = `
        <div class="msg-avatar">${avatarLabel === 'You' ? '👤' : '🏛️'}</div>
        <div class="msg-body">
            <div class="msg-bubble">${formatMessageHtml(content)}</div>
        </div>
    `;
    container.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
}

function createStreamingMessage() {
    hideWelcome();
    const container = DOM.messages();

    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
        <div class="msg-avatar">🏛️</div>
        <div class="msg-body">
            <div class="msg-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    container.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
}

function scrollToBottom() {
    const container = DOM.chatContainer();
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

function applyAssistantMeta(body, data) {
    body.insertAdjacentHTML('beforeend', buildGroundingMetaHtml(data));
    const evidenceHtml = buildEvidenceHtml(data);
    if (evidenceHtml) {
        body.insertAdjacentHTML('beforeend', evidenceHtml);
    }
}

async function sendMessage(question) {
    appendMessage('user', question);
    chatHistory.push({ role: 'user', content: question });
    const msgEl = createStreamingMessage();
    const bubble = msgEl.querySelector('.msg-bubble');
    const body = msgEl.querySelector('.msg-body');

    const lang = DOM.langSelect().value;
    const historySlice = chatHistory.slice(-10);

    try {
        const headers = { 'Content-Type': 'application/json' };
        const virtualAssistantToken = getVirtualAssistantToken();
        if (virtualAssistantToken) {
            headers.Authorization = `Bearer ${virtualAssistantToken}`;
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify({ question, language: lang, history: historySlice }),
        });

        if (!response.ok) {
            let errorMessage = `Request failed (${response.status})`;

            try {
                const data = await response.json();
                errorMessage = data.error || errorMessage;
                if (data.retry_after_seconds) {
                    errorMessage = `${errorMessage} Retry after ${data.retry_after_seconds}s.`;
                }
            } catch {
                // Ignore JSON parse errors and keep the fallback message.
            }

            bubble.textContent = errorMessage;
            bubble.style.color = 'oklch(0.55 0.20 25)';
            scrollToBottom();
            return;
        }

        if (!response.body) {
            throw new Error('Empty response body');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';
        let finalMeta = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6).trim();
                if (!raw) continue;

                try {
                    const data = JSON.parse(raw);

                    if (data.type === 'answer') {
                        fullText = data.content || '';
                        const normalized = normalizeAssistantText(fullText, data.language || finalMeta?.language);
                        fullText = normalized.text;
                        bubble.innerHTML = formatMessageHtml(fullText);
                        scrollToBottom();
                    }

                    if (data.type === 'token') {
                        fullText += data.content;
                        const normalized = normalizeAssistantText(fullText, finalMeta?.language);
                        bubble.innerHTML = formatMessageHtml(normalized.text);
                        scrollToBottom();
                    }

                    if (data.type === 'done') {
                        finalMeta = data;
                        if (!fullText && data.answer) {
                            const normalized = normalizeAssistantText(data.answer, data.language);
                            fullText = normalized.text;
                            bubble.innerHTML = formatMessageHtml(fullText);
                        }
                        applyAssistantMeta(body, data);
                    }

                    if (data.type === 'error') {
                        bubble.textContent = `Error: ${data.content}`;
                        bubble.style.color = 'oklch(0.55 0.20 25)';
                    }
                } catch {
                    // Ignore malformed chunks.
                }
            }
        }

        if (!fullText) {
            bubble.textContent = getUiStrings(finalMeta?.language).noResponse;
        } else {
            chatHistory.push({ role: 'assistant', content: fullText });
        }
    } catch (err) {
        bubble.textContent = `Connection error: ${err.message}`;
        bubble.style.color = 'oklch(0.55 0.20 25)';
    }

    scrollToBottom();
    DOM.btnSend().disabled = !DOM.chatInput().value.trim();
}

function sendHint(el) {
    const text = el.textContent;
    DOM.chatInput().value = text;
    DOM.btnSend().disabled = false;
    DOM.chatForm().requestSubmit();
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        DOM.statsMini().innerHTML =
            `📊 ${data.total_vectors} vectors<br>🧠 ${data.embedding_model.split('/').pop()}`;
    } catch {
        DOM.statsMini().textContent = '';
    }
}

if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        initVirtualAssistantToken();
        initNavigation();
        initChatInput();
        initUploadPanel();
        loadStats();
    });
}

globalThis.__appTest = {
    buildEvidenceHtml,
    buildGroundingMetaHtml,
    escapeHtml,
    formatMessageHtml,
    getUiStrings,
    normalizeAssistantText,
    tryParseAssistantPayload,
};

if (typeof window !== 'undefined') {
    window.sendHint = sendHint;
}
