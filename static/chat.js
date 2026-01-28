/**
 * 对话功能 JavaScript - 完整版
 * 包含：WebSocket 流式对话、历史记录侧边栏加载、思考过程展示
 */

const WS_URL = 'ws://localhost:8000/ws/stream';
const API_URL = 'http://localhost:8000/api';

let ws = null;
let isConnected = false;
let isProcessing = false;
let currentThinkingContainer = null;
let currentStreamingAnswer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// DOM 元素引用
const messagesWrapper = document.getElementById('messagesWrapper');
const messagesArea = document.getElementById('messagesArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const historyList = document.getElementById('historyList'); // 侧边栏列表
const newChatBtn = document.getElementById('newChatBtn'); // 顶部新建按钮
const sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn'); // 侧边栏新建按钮
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// ============ 1. 历史记录加载与管理 ============
async function loadSavedHistory() {
    try {
        console.log("正在加载历史记录...");
        const response = await fetch(`${API_URL}/history/saved`);
        
        if (!response.ok) {
            console.warn("无法连接到历史记录接口");
            return;
        }

        const data = await response.json();
        
        // 清空列表
        if (historyList) {
            historyList.innerHTML = '';
        }

        // ★★★ 适配新的数据结构：sessions ★★★
        if (data.success && Array.isArray(data.sessions) && data.sessions.length > 0) {
            // 按更新时间倒序排列，最新的显示在最上面
            const sortedSessions = [...data.sessions].sort((a, b) => 
                new Date(b.updated_at) - new Date(a.updated_at)
            );

            sortedSessions.forEach((session) => {
                const li = document.createElement('li');
                li.className = 'history-item';
                
                // 使用 session 的 title，如果没有则用第一条对话的内容
                let title = session.title || '新对话';
                
                // 如果 title 是 "新对话" 且有对话内容，用第一条用户消息作为标题
                if (title === '新对话' && session.conversation && session.conversation.length > 0) {
                    const firstMsg = session.conversation[0].user_content;
                    if (firstMsg) {
                        title = firstMsg.length > 20 
                            ? firstMsg.substring(0, 20) + '...' 
                            : firstMsg;
                    }
                }
                
                li.textContent = title;
                
                // 显示对话数量和时间
                const conversationInfo = document.createElement('span');
                conversationInfo.className = 'conversation-info';
                conversationInfo.textContent = ` (${session.conversation_count}条)`;
                conversationInfo.style.fontSize = '0.85em';
                conversationInfo.style.color = '#999';
                li.appendChild(conversationInfo);
                
                // 完整标题作为 tooltip
                li.title = `${title}\n对话数: ${session.conversation_count}\n时间: ${new Date(session.updated_at).toLocaleString('zh-CN')}`;
                
                // 点击事件：恢复整个 session 的对话
                li.onclick = () => restoreSession(session);
                
                historyList.appendChild(li);
            });
        } else {
            // 没有历史记录时显示提示
            const emptyTip = document.createElement('li');
            emptyTip.className = 'history-empty';
            emptyTip.textContent = '暂无历史记录';
            emptyTip.style.textAlign = 'center';
            emptyTip.style.color = '#999';
            emptyTip.style.padding = '20px';
            historyList.appendChild(emptyTip);
        }
    } catch (error) {
        console.error("加载历史记录失败:", error);
    }
}

/**
 * 恢复显示某一段历史对话
 */
function restoreSession(session) {
    // 1. 清空当前屏幕
    resetChatUI();

    // 2. 遍历显示所有对话
    if (session.conversation && session.conversation.length > 0) {
        session.conversation.forEach(msg => {
            // 显示用户提问
            if (msg.user_content) {
                addMessage('user', msg.user_content);
            }
            // 显示 AI 回答
            if (msg.ai_content) {
                addMessage('assistant', msg.ai_content);
            }
        });
    }
}
/**
 * 重置聊天界面 (清空消息，显示欢迎页)
 * 但这里我们实际上是清空消息，隐藏欢迎页(如果有新消息)
 */
function resetChatUI() {
    messagesWrapper.innerHTML = '';
    // 隐藏欢迎页 (因为要显示消息了)
    hideWelcomeScreen();
    // 重置状态
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
}

/**
 * 完全重置为初始状态 (点击新建对话时)
 */
function startNewChat() {
    messagesWrapper.innerHTML = '';
    // 重新把欢迎页放回去
    messagesWrapper.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';
    
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    messageInput.value = '';
    messageInput.focus();
}

// ============ 2. WebSocket 连接管理 ============

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('✅ WebSocket 连接成功');
        isConnected = true;
        reconnectAttempts = 0;
        updateStatus(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket 错误:', error);
        updateStatus(false);
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket 连接关闭');
        isConnected = false;
        updateStatus(false);
        
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            setTimeout(connectWebSocket, delay);
        }
    };
}

function updateStatus(connected) {
    if (statusIndicator) {
        if (connected) {
            statusIndicator.className = 'status-indicator connected';
            if (statusText) statusText.textContent = '已连接';
        } else {
            statusIndicator.className = 'status-indicator disconnected';
            if (statusText) statusText.textContent = '未连接';
        }
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'step':
            handleStepUpdate(data);
            break;
        case 'token':
            handleTokenUpdate(data.content);
            break;
        case 'done':
            handleDone();
            break;
        case 'error':
            handleError(data.message);
            break;
    }
}

// ============ 3. 消息渲染与流式处理 ============

function hideWelcomeScreen() {
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
}

function addMessage(role, content) {
    hideWelcomeScreen();
    
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    
    let innerHTML = '';
    
    if (role === 'user') {
        // 用户消息，简单文本转义
        const textDiv = document.createElement('div');
        textDiv.textContent = content;
        innerHTML = `
            <div class="message-header">
                <div class="message-avatar">👤</div>
                <div class="message-author">你</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content">${textDiv.innerHTML}</div>
        `;
    } else {
        // AI 消息，Markdown 解析
        const parsed = typeof marked !== 'undefined' ? marked.parse(content) : content;
        innerHTML = `
            <div class="message-header">
                <div class="message-avatar">🤖</div>
                <div class="message-author">AI 助手</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content">${parsed}</div>
        `;
    }
    
    div.innerHTML = innerHTML;
    messagesWrapper.appendChild(div);
    scrollToBottom();
}

// 处理思考过程 (Step)
function handleStepUpdate(data) {
    hideWelcomeScreen();

    // 如果还没有思考容器，创建一个
    if (!currentThinkingContainer) {
        currentThinkingContainer = document.createElement('div');
        currentThinkingContainer.className = 'thinking-process';
        currentThinkingContainer.innerHTML = `
            <div class="thinking-header" onclick="toggleThinking(this)">
                <span class="thinking-toggle">▼</span>
                <span class="thinking-title">思考过程</span>
                <span class="thinking-icon">⚙️</span>
            </div>
            <div class="thinking-content"></div>
        `;
        messagesWrapper.appendChild(currentThinkingContainer);
    }

    const stepsContainer = currentThinkingContainer.querySelector('.thinking-content');
    
    const stepDiv = document.createElement('div');
    stepDiv.className = `thinking-step ${getStepClass(data.step)}`;
    
    const icon = getStepIcon(data.step);
    const title = data.title || '处理中';
    const description = data.description || '';
    
    stepDiv.innerHTML = `
        <span class="step-icon">${icon}</span>
        <div class="step-content">
            <div class="step-title">${title}</div>
            <div class="step-description">${description}</div>
        </div>
    `;
    
    stepsContainer.appendChild(stepDiv);
    scrollToBottom();
}

function getStepIcon(step) {
    const s = step.toLowerCase();
    if (s.includes('search')) return '🔍';
    if (s.includes('doc')) return '📚';
    if (s.includes('plan')) return '🤔';
    if (s.includes('chat')) return '💬';
    return '⚙️';
}

function getStepClass(step) {
    const s = step.toLowerCase();
    if (s.includes('analyzing')) return 'analyzing';
    if (s.includes('plan')) return 'planning';
    if (s.includes('chat')) return 'chatting';
    return '';
}

// 处理文本流 (Token)
function handleTokenUpdate(token) {
    if (!currentStreamingAnswer) {
        // 创建新的 AI 回复框
        currentStreamingAnswer = document.createElement('div');
        currentStreamingAnswer.className = 'message assistant';
        const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        
        currentStreamingAnswer.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">🤖</div>
                <div class="message-author">AI 助手</div>
                <div class="message-time">${time}</div>
            </div>
            <div class="message-content streaming" data-raw=""></div>
        `;
        messagesWrapper.appendChild(currentStreamingAnswer);
    }
    
    const contentDiv = currentStreamingAnswer.querySelector('.message-content');
    // 获取当前暂存的原始文本
    const currentRaw = contentDiv.dataset.raw || '';
    const newRaw = currentRaw + token;
    contentDiv.dataset.raw = newRaw;
    
    // 实时解析 Markdown
    if (typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(newRaw);
    } else {
        contentDiv.textContent = newRaw;
    }
    
    scrollToBottom();
}

// 处理完成 (Done)
function handleDone() {
    if (currentStreamingAnswer) {
        const contentDiv = currentStreamingAnswer.querySelector('.message-content');
        contentDiv.classList.remove('streaming');
    }
    
    currentThinkingContainer = null;
    currentStreamingAnswer = null;
    isProcessing = false;
    
    // 恢复输入框
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
    if (messageInput) messageInput.focus();

    // ★★★ 对话结束后，重新加载历史记录，确保刚才的对话出现在侧边栏 ★★★
    loadSavedHistory();
}

// 处理错误 (Error)
function handleError(msg) {
    addMessage('assistant', `❌ 错误: ${msg}`);
    isProcessing = false;
    if (sendButton) sendButton.disabled = false;
    if (messageInput) messageInput.disabled = false;
}

// ============ 4. 发送与交互逻辑 ============

function sendMessage(text = null) {
    const message = text || messageInput.value.trim();
    
    if (!message || !isConnected || isProcessing) {
        if (!isConnected) showToast("未连接到服务器", "error");
        return;
    }

    // 1. 显示用户消息
    addMessage('user', message);
    
    // 2. 发送 WebSocket
    ws.send(JSON.stringify({ message }));
    
    // 3. UI 状态更新
    if (!text) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
    
    isProcessing = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        if (messagesArea) {
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }
    });
}

function autoResizeTextarea() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
}

function attachQuickPromptListeners() {
    const cards = document.querySelectorAll('.quick-prompt-card');
    cards.forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            sendMessage(prompt);
        });
    });
}

// ============ 5. 初始化绑定 ============

// 绑定发送按钮
if (sendButton) sendButton.addEventListener('click', () => sendMessage());

// 绑定输入框回车
if (messageInput) {
    messageInput.addEventListener('input', autoResizeTextarea);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// 绑定新建对话按钮 (Header 和 侧边栏)
if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);
if (sidebarNewChatBtn) sidebarNewChatBtn.addEventListener('click', startNewChat);

// 启动
attachQuickPromptListeners();
connectWebSocket();
loadSavedHistory(); // 页面加载时自动获取历史记录