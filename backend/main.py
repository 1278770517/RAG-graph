"""
FastAPI 后端 - 提供聊天 API
使用 graph.stream() 返回流式响应（同步流式，非异步）
"""
import json
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from torch.distributed.tensor.parallel.input_reshard import input_reshard

from backend import state_manager
from state_manager import SessionManager
from utils.embeding_uiliti import image_to_base64
from workflow.Workflow1 import  active_graph, update_state


# ==================== Pydantic 模型 ====================
class ChatRequest(BaseModel):
    """发送消息请求"""
    content: str = Field(..., description="用户消息内容")


class ChatResponse(BaseModel):
    """对话响应（非流式，用于列表等）"""
    thread_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class ThreadCreateResponse(BaseModel):
    """创建对话响应"""
    thread_id: str
    title: str


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: str
    type: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class ThreadDetailResponse(BaseModel):
    """对话详情响应"""
    thread_id: str
    title: str
    messages: List[MessageItem]


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Chat API",
    description="FastAPI + LangGraph 聊天后端",
    version="1.0.0"
)
sm=SessionManager()
# CORS - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 路由 ====================

@app.post("/api/chat/new", response_model=ThreadCreateResponse)
def create_chat():
    """
    新建对话
    返回 thread_id 和默认标题
    """
    thread_id = str(uuid.uuid4())
    return ThreadCreateResponse(
        thread_id=thread_id,
        title=f"对话{thread_id}",
    )



@app.delete("/api/chat/{thread_id}")
def delete_chat(thread_id: str):
    graph=active_graph()
    """
    删除对话
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    sm.connect()
    checkpoint, history =sm.get_memory()
    try:
        checkpoint.delete_thread(thread_id)
    except Exception as e:
        return {"status": "404", "message": f"{e}对话不存在"}


    return {"status": "ok", "message": "对话已删除"}


@app.post("/api/chat/{thread_id}")
def chat_stream(thread_id: str, request: ChatRequest):
    """
    发送消息，返回 SSE 流式响应
    使用 graph.invoke() 配合循环来处理中断
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    graphs = active_graph()

    def event_generator():
        # 1. 准备输入数据
        image_base64 = None
        text = None

        # 解析图片和文本
        if '&' in request.content:
            text = request.content.split('&')[0]
            image_path = request.content.split('&')[1]
            if image_path and os.path.isfile(image_path):
                image_base64 = {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image_path)[0]},
                }
        elif os.path.isfile(request.content):
            image_base64 = {
                "type": "image_url",
                "image_url": {"url": image_to_base64(request.content)[0]},
            }
        else:
            text = request.content

        # 构造 HumanMessage
        message_content = []
        if text:
            message_content.append({'type': 'text', 'text': text})
        if image_base64:
            message_content.append(image_base64)

        human_msg = HumanMessage(content=message_content)

        # 2. 初始 inputs
        inputs = {"messages": [human_msg]}

        # 如果当前状态已经在等待中断（即 state.next 不为空），
        # 说明这是用户点击“批准”或“拒绝”后的后续请求，此时 inputs 应该为 None，让图继续跑
        current_state = graphs.get_state(config)
        if current_state.next:
            print(f"检测到流程处于中断/挂起状态，准备继续执行。上一步节点: {current_state.next}")
            update_state(request.content,config)
            inputs = None

        # 3. 【核心】循环处理逻辑
        while True:
            # 执行一步 (遇到 interrupt 会自动停在这里)
            current_state = graphs.invoke(inputs, config)

            # 重置 inputs，因为后续继续执行不需要再重复传入初始消息
            inputs = None

            # 4. 获取最新产生的消息
            # 原理：比较 current_state.messages 的长度和调用前的长度（这里简化处理，直接取最后一条）
            # 更好的做法是：LangGraph 的 invoke 返回的 state 中包含了所有历史，
            # 我们只需要处理“在这一轮 invoke 中新增的消息”。
            # 但为了简化，我们直接取最新的一条 AI 消息进行发送。
            # 注意：为了避免重复发送，你需要记录上次发送到的位置，或者简单点：
            # 只要这轮 invoke 产生了新的 AI 输出，就发出去。

            messages = current_state.get("messages", [])
            if not messages:
                continue

            # 获取最新的消息（通常是我们刚刚生成的）
            last_message = messages[-1]

            if isinstance(last_message, AIMessage):
                # 发送文本
                if last_message.content:
                    yield sse_payload("message", {"data": last_message.content})

                # 发送工具调用
                elif last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        yield sse_payload("tool_call", {
                            "name": tool_call.get("name"),
                            "args": tool_call.get("args"),
                        })

            # 5. 检查是否中断或结束
            # graphs.get_state(config) 获取最新状态信息
            state_info = graphs.get_state(config)

            # 如果 state_info.next 为空，说明流程彻底结束
            if not state_info.next:
                yield f"data: {json.dumps({'event': 'done'}, ensure_ascii=False)}\n\n"
                break  # 退出循环，结束 SSE 连接

            # 如果 state_info.next 不为空，说明遇到了 interrupt 或者图还没跑完
            # 此时我们需要 yield 一个中断信号给前端，然后暂停循环（等待前端再次调用接口）
            if state_info.next:
                yield sse_payload("interrupt", {
                    "data": (
                        "由于系统自我评估后，发现AI的回复不是非常准确，您是否认可以下输出？\n"
                        "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！"
                    )
                })
                break  # 暂停循环，保持连接或断开，等待用户操作后的下一次请求

    def sse_payload(event: str, payload: dict) -> str:
        """统一包装 SSE 事件"""
        return f"data: {json.dumps({'event': event, **payload}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )

@app.get("/api/health")
def health_check():
    """健康检查"""
    return {"status": "ok"}


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
