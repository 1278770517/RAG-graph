const BASE_URL = '';

export const chatApi = {
  async health() {
    const res = await fetch(`${BASE_URL}/api/health`);
    return res.json();
  },

  async createThread() {
    const res = await fetch(`${BASE_URL}/api/chat/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error('创建对话失败');
    return res.json();
  },

  async deleteThread(threadId) {
    const res = await fetch(`${BASE_URL}/api/chat/${threadId}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.message || '删除失败');
    }
    return res.json();
  },

  sendMessage(threadId, content) {
    return fetch(`${BASE_URL}/api/chat/${threadId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
  },
};
