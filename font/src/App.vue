<template>
  <div class="chat-app">
    <!-- 左侧边栏 -->
    <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-header">
        <button class="new-chat-btn" @click="createNewThread" :disabled="loading">
          <span class="icon">+</span>
          <span>新建对话</span>
        </button>
        <button class="toggle-btn" @click="sidebarCollapsed = !sidebarCollapsed">
          {{ sidebarCollapsed ? '→' : '←' }}
        </button>
      </div>

      <div class="thread-list">
        <div
          v-for="thread in threads"
          :key="thread.thread_id"
          class="thread-item"
          :class="{ active: currentThreadId === thread.thread_id }"
          @click="switchThread(thread.thread_id)"
        >
          <span class="thread-title">{{ thread.title }}</span>
          <button
            class="delete-btn"
            @click.stop="deleteThread(thread.thread_id)"
            :disabled="deletingId === thread.thread_id"
          >
            <span v-if="deletingId === thread.thread_id">...</span>
            <span v-else>×</span>
          </button>
        </div>
        <div v-if="threads.length === 0" class="empty-threads">
          暂无对话，点击上方新建
        </div>
      </div>
    </aside>

    <!-- 主聊天区 -->
    <main class="chat-main">
      <div v-if="!currentThreadId" class="welcome-screen">
        <h1>🤖 AI 助手</h1>
        <p>请从左侧选择一个对话或创建新对话开始聊天</p>
      </div>

      <template v-else>
        <!-- 消息列表 -->
        <div class="messages-container" ref="messagesContainer">
          <div
            v-for="msg in currentMessages"
            :key="msg.id"
            class="message-wrapper"
            :class="msg.role"
          >
            <div class="avatar">
              {{ msg.role === 'user' ? '👤' : '🤖' }}
            </div>
            <div class="message-content">
              <!-- 文本内容 -->
              <div v-if="msg.text" class="text-content" v-html="renderMarkdown(msg.text)"></div>

              <!-- 图片内容 -->
              <div v-if="msg.imageUrl" class="image-content">
                <img :src="msg.imageUrl" alt="uploaded" />
              </div>

              <!-- 工具调用 -->
              <div v-if="msg.toolCalls?.length" class="tool-calls">
                <div v-for="(tool, ti) in msg.toolCalls" :key="ti" class="tool-call">
                  <div class="tool-name">🔧 {{ tool.name }}</div>
                  <pre class="tool-args">{{ JSON.stringify(tool.args, null, 2) }}</pre>
                </div>
              </div>

              <!-- 中断提示 -->
              <div v-if="msg.isInterrupt" class="interrupt-box">
                <div class="interrupt-text">{{ msg.text }}</div>
                <div class="interrupt-actions">
                  <button @click="handleInterrupt('approve')" class="btn-approve">✅ 认可</button>
                  <button @click="handleInterrupt('rejected')" class="btn-reject">❌ 重新生成</button>
                </div>
              </div>
            </div>
          </div>

          <!-- 流式输出中的光标 -->
          <div v-if="streaming" class="message-wrapper ai streaming">
            <div class="avatar">🤖</div>
            <div class="message-content">
              <span class="cursor">▌</span>
            </div>
          </div>
        </div>

        <!-- 输入区域 -->
        <div class="input-area">
          <div v-if="interruptMode" class="interrupt-hint">
            请确认上方 AI 的输出，或选择重新生成
          </div>
          <div class="input-box">
            <textarea
              v-model="inputMessage"
              :disabled="streaming || interruptMode"
              :placeholder="interruptMode ? '请先确认上方输出...' : '输入消息，支持 & 分隔图片路径'"
              @keydown.enter.prevent="handleEnter"
              rows="1"
              ref="textareaRef"
            ></textarea>
            <button
              class="send-btn"
              @click="sendMessage"
              :disabled="!inputMessage.trim() || streaming || interruptMode"
            >
              {{ streaming ? '...' : '➤' }}
            </button>
          </div>
        </div>
      </template>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, watch } from 'vue';
import { chatApi } from './api/chat.js';

// ==================== 工具函数 ====================
let _msgId = 0;
function genId() {
  return 'msg_' + Date.now() + '_' + (++_msgId);
}

// ==================== 状态 ====================
const threads = ref([]);
const currentThreadId = ref(null);
const messagesMap = ref({});        // { threadId: [msg1, msg2, ...] }
const inputMessage = ref('');
const loading = ref(false);
const deletingId = ref(null);
const streaming = ref(false);
const interruptMode = ref(false);
const sidebarCollapsed = ref(false);
const messagesContainer = ref(null);

// 核心修复：用消息 ID 精确追踪当前正在接收的 AI 消息，绝不依赖数组索引
const currentAiMsgId = ref(null);

// ==================== 计算属性 ====================
const currentMessages = computed(() => {
  return messagesMap.value[currentThreadId.value] || [];
});

// ==================== 消息操作（每次创建新数组，确保 Vue 响应式必触发）====================

/** 添加消息，返回消息 id */
function addMessage(threadId, message) {
  const msg = { ...message, id: genId() };
  const current = messagesMap.value[threadId] || [];
  messagesMap.value[threadId] = [...current, msg];
  return msg.id;
}

/** 通过 id 更新消息中的指定字段 */
function updateMessageById(threadId, msgId, patch) {
  const current = messagesMap.value[threadId] || [];
  const idx = current.findIndex(m => m.id === msgId);
  if (idx === -1) return;
  messagesMap.value[threadId] = [
    ...current.slice(0, idx),
    { ...current[idx], ...patch },
    ...current.slice(idx + 1),
  ];
}

/** 删除指定 id 的消息 */
function removeMessageById(threadId, msgId) {
  const current = messagesMap.value[threadId] || [];
  messagesMap.value[threadId] = current.filter(m => m.id !== msgId);
}

// ==================== 对话管理 ====================

async function createNewThread() {
  loading.value = true;
  try {
    const data = await chatApi.createThread();
    threads.value.unshift({
      thread_id: data.thread_id,
      title: data.title,
      created_at: Date.now(),
    });
    messagesMap.value[data.thread_id] = [];
    currentThreadId.value = data.thread_id;
    inputMessage.value = '';
  } catch (err) {
    alert('创建对话失败: ' + err.message);
  } finally {
    loading.value = false;
  }
}

function switchThread(threadId) {
  currentThreadId.value = threadId;
  inputMessage.value = '';
  interruptMode.value = false;
  currentAiMsgId.value = null;
  nextTick(() => scrollToBottom());
}

async function deleteThread(threadId) {
  if (!confirm('确定要删除这个对话吗？')) return;
  deletingId.value = threadId;
  try {
    await chatApi.deleteThread(threadId);
    threads.value = threads.value.filter(t => t.thread_id !== threadId);
    delete messagesMap.value[threadId];
    if (currentThreadId.value === threadId) {
      currentThreadId.value = threads.value[0]?.thread_id || null;
    }
  } catch (err) {
    alert('删除失败: ' + err.message);
  } finally {
    deletingId.value = null;
  }
}

// ==================== 发送消息 & SSE 解析 ====================

async function sendMessage() {
  const content = inputMessage.value.trim();
  if (!content || !currentThreadId.value || streaming.value) return;

  const threadId = currentThreadId.value;

  // 1. 添加用户消息
  const userMsg = parseUserMessage(content);
  addMessage(threadId, { role: 'user', text: userMsg.text, imageUrl: userMsg.imageUrl });
  inputMessage.value = '';
  streaming.value = true;
  interruptMode.value = false;
  currentAiMsgId.value = null;

  // 2. 添加 AI 占位消息，并记录其 id
  const aiMsgId = addMessage(threadId, { role: 'ai', text: '', toolCalls: [] });
  currentAiMsgId.value = aiMsgId;

  try {
    const response = await chatApi.sendMessage(threadId, content);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const dataStr = trimmed.slice(6).trim();
        if (dataStr === '[DONE]') continue;

        try {
          const event = JSON.parse(dataStr);
          handleSSEEvent(event, threadId);
        } catch (e) {
          console.warn('解析 SSE 数据失败:', trimmed, e);
        }
      }
    }
  } catch (err) {
    updateMessageById(threadId, aiMsgId, { text: `\n\n[错误: ${err.message}]` });
  } finally {
    streaming.value = false;
    currentAiMsgId.value = null;
    nextTick(() => scrollToBottom());
  }
}

/** 解析用户消息（支持 & 分隔图片路径） */
function parseUserMessage(content) {
  const result = { text: '', imageUrl: null };
  if (content.includes('&')) {
    const parts = content.split('&');
    result.text = parts[0].trim();
    result.imageUrl = parts.slice(1).join('&').trim();
  } else {
    result.text = content;
  }
  return result;
}

/** 处理 SSE 事件（核心修复：通过 currentAiMsgId 精确更新，绝不碰其他消息） */
function handleSSEEvent(event, threadId) {
  const aiMsgId = currentAiMsgId.value;
  if (!aiMsgId) return;

  switch (event.event) {
    case 'message': {
      const text = typeof event.data === 'string' ? event.data : JSON.stringify(event.data);
      if (!text) break;

      const current = messagesMap.value[threadId] || [];
      const msg = current.find(m => m.id === aiMsgId);
      if (!msg) break;

      // 追加文本（假设后端发送增量；如果后端发送完整内容，请改为直接赋值）
      updateMessageById(threadId, aiMsgId, { text: msg.text + text });
      nextTick(() => scrollToBottom());
      break;
    }

    case 'tool_call': {
      const current = messagesMap.value[threadId] || [];
      const msg = current.find(m => m.id === aiMsgId);
      if (!msg) break;

      updateMessageById(threadId, aiMsgId, {
        toolCalls: [...(msg.toolCalls || []), { name: event.name, args: event.args }],
      });
      break;
    }

    case 'interrupt': {
      // 核心修复：不再替换当前 AI 消息，而是追加一条独立的 interrupt 消息
      // 这样旧 AI 回复完整保留，interrupt 作为新消息出现
      interruptMode.value = true;
      streaming.value = false;
      currentAiMsgId.value = null;
      addMessage(threadId, {
        role: 'ai',
        text: event.data,
        isInterrupt: true,
      });
      break;
    }

    case 'done': {
      streaming.value = false;
      currentAiMsgId.value = null;
      break;
    }

    case 'error': {
      const current = messagesMap.value[threadId] || [];
      const msg = current.find(m => m.id === aiMsgId);
      if (msg) {
        updateMessageById(threadId, aiMsgId, { text: msg.text + `\n\n[错误: ${event.error}]` });
      }
      streaming.value = false;
      currentAiMsgId.value = null;
      break;
    }

    default:
      console.log('未知 SSE 事件:', event);
  }
}

/** 处理中断确认 */
function handleInterrupt(action) {
  interruptMode.value = false;
  // 删除所有 interrupt 消息（保留所有历史 AI 回复）
  const current = messagesMap.value[currentThreadId.value] || [];
  messagesMap.value[currentThreadId.value] = current.filter(m => !m.isInterrupt);

  inputMessage.value = action;
  sendMessage();
}

// ==================== 输入 & 滚动 & 渲染 ====================

function handleEnter(e) {
  if (!e.shiftKey) {
    sendMessage();
  }
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
}

function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

watch(currentMessages, () => nextTick(scrollToBottom), { deep: true });
</script>

<style scoped>
/* ==================== 布局 ==================== */
.chat-app {
  display: flex;
  height: 100vh;
  width: 100vw;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
  overflow: hidden;
}

/* ==================== 侧边栏 ==================== */
.sidebar {
  width: 260px;
  background: #202123;
  color: #fff;
  display: flex;
  flex-direction: column;
  transition: width 0.3s;
  flex-shrink: 0;
}

.sidebar.collapsed {
  width: 0;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #444;
  display: flex;
  gap: 8px;
  align-items: center;
}

.new-chat-btn {
  flex: 1;
  padding: 10px 14px;
  background: transparent;
  border: 1px solid #444;
  color: #fff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: background 0.2s;
}

.new-chat-btn:hover {
  background: #2d2d30;
}

.new-chat-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.toggle-btn {
  padding: 8px;
  background: transparent;
  border: none;
  color: #fff;
  cursor: pointer;
  font-size: 16px;
}

.thread-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.thread-item {
  padding: 10px 12px;
  margin-bottom: 4px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.2s;
  font-size: 14px;
}

.thread-item:hover {
  background: #2d2d30;
}

.thread-item.active {
  background: #343541;
}

.thread-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-right: 8px;
}

.delete-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  font-size: 18px;
  padding: 0 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.thread-item:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  color: #ff4444;
}

.empty-threads {
  text-align: center;
  color: #888;
  padding: 40px 20px;
  font-size: 13px;
}

/* ==================== 主聊天区 ==================== */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #fff;
  overflow: hidden;
  min-width: 0;
}

.welcome-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
}

.welcome-screen h1 {
  font-size: 48px;
  margin-bottom: 16px;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ==================== 消息气泡 ==================== */
.message-wrapper {
  display: flex;
  gap: 12px;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-wrapper.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
  background: #e3e3e3;
}

.message-wrapper.ai .avatar {
  background: #10a37f;
}

.message-content {
  max-width: calc(100% - 60px);
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  font-size: 15px;
  word-break: break-word;
}

.user .message-content {
  background: #2d2d30;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.ai .message-content {
  background: #f7f7f8;
  color: #333;
  border-bottom-left-radius: 4px;
}

.text-content :deep(pre) {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
}

.text-content :deep(code) {
  background: #f0f0f0;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Consolas', monospace;
  font-size: 13px;
}

.text-content :deep(pre code) {
  background: none;
  padding: 0;
}

.image-content img {
  max-width: 300px;
  max-height: 300px;
  border-radius: 8px;
  margin-top: 8px;
}

/* 流式光标 */
.cursor {
  display: inline-block;
  color: #10a37f;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* 工具调用 */
.tool-calls {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e0e0e0;
}

.tool-call {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 8px 12px;
  margin-top: 6px;
}

.tool-name {
  font-weight: 600;
  color: #10a37f;
  font-size: 13px;
  margin-bottom: 4px;
}

.tool-args {
  background: #f5f5f5;
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  overflow-x: auto;
  margin: 0;
}

/* 中断提示 */
.interrupt-box {
  background: #fff8e1;
  border: 1px solid #ffc107;
  border-radius: 8px;
  padding: 12px;
}

.interrupt-text {
  color: #5d4037;
  margin-bottom: 12px;
  white-space: pre-wrap;
}

.interrupt-actions {
  display: flex;
  gap: 10px;
}

.btn-approve, .btn-reject {
  padding: 8px 16px;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: opacity 0.2s;
}

.btn-approve {
  background: #10a37f;
  color: white;
}

.btn-reject {
  background: #ff4444;
  color: white;
}

.btn-approve:hover, .btn-reject:hover {
  opacity: 0.9;
}

/* ==================== 输入区域 ==================== */
.input-area {
  border-top: 1px solid #e0e0e0;
  padding: 16px 20px;
  background: #fff;
  flex-shrink: 0;
}

.interrupt-hint {
  text-align: center;
  color: #ff9800;
  font-size: 13px;
  margin-bottom: 8px;
}

.input-box {
  display: flex;
  gap: 10px;
  max-width: 900px;
  margin: 0 auto;
  background: #f7f7f8;
  border-radius: 12px;
  padding: 4px;
  border: 1px solid #e0e0e0;
}

.input-box:focus-within {
  border-color: #10a37f;
}

textarea {
  flex: 1;
  border: none;
  background: transparent;
  padding: 12px;
  font-size: 15px;
  resize: none;
  outline: none;
  max-height: 200px;
  font-family: inherit;
}

.send-btn {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  border: none;
  background: #10a37f;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  align-self: flex-end;
  margin: 4px;
  transition: background 0.2s;
}

.send-btn:hover:not(:disabled) {
  background: #0d8c6d;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* 响应式 */
@media (max-width: 768px) {
  .sidebar {
    position: absolute;
    z-index: 100;
    height: 100%;
  }

  .message-wrapper {
    max-width: 100%;
    padding: 0 10px;
  }
}
</style>
