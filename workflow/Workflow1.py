# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import uuid
import asyncio
import sys
from typing import List, Dict
import gradio as gr
from gradio import ChatMessage
from ragas.messages import ToolMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.postgres import PostgresStore, AsyncPostgresStore
from llm import LLM_QWEN
from utils.common_utils import draw_graph
from utils.embeding_uiliti import image_to_base64
from utils.log_utils import log
from workflow.alll_route import route_only_image, route_evaluate_node, route_human_node, route_human_approval_node, \
    route_llm_or_retriever
from workflow.envaluate_node import answer_envaluate
from workflow.print_messages import pretty_print_messages
from workflow.save_context import get_milvus_writer
from workflow.search_node import SearchCOntextNode, retriever_node
from workflow.state import MultiModalRAGState
from workflow.tools import search_context, web_search
from backend.state_manager import SessionManager
S_M=SessionManager()
tool=[search_context]
web_tools=[web_search]


def InvalidInputError(param):
    pass


def process_inpute(state:MultiModalRAGState,config:RunnableConfig):
    print(f'STATE:{state}')
    user_name=config['configurable'].get('user_name','Constantinus')
    last_message=state['messages'][-1]
    log.info(f"用户 {user_name} 输入：{last_message}")
    input_type = 'has_text'
    text_content = None
    image_url = None
    if isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,list):
            for item in last_message.content:
                if item.get('type')=='text':
                    text_content = item.get('text')
                elif item.get('type')=='image_url':
                    image_url = item.get('image_url','').get('url')
                    if image_url:
                        url=image_url
        else:
            raise InvalidInputError(f"用户输入的消息错误！原始输入：{last_message}")
    if not text_content and image_url:
        input_type='only_image'
    return {"input_type": input_type, 'user_id': user_name, 'inpute_context': text_content, 'inpute_image': image_url}
def frist_chatbot(state:MultiModalRAGState):
    print(f'STATE:{state["input_type"]}')
    # llm_with_tools = llm.bind_tools(tools)
    llm_with_tools =LLM_QWEN.bind_tools(tool)
    # # 修改后的系统提示词示例
    system_message = SystemMessage(content="""你是一名知识库 的AI助手。你的核心任务是处理用户关于文档。

        # 核心指令（必须严格遵守）：
        1.  **首要规则**：当用户问问题时候，你**必须且先**调用 `search_context` 工具来获取信息。
        2.  **禁止行为**：你的回答必须完全基于工具返回的知识库内容。
        3.  **兜底策略**：如果工具返回了相关信息，请基于这些信息组织答案。如果工具明确返回“未找到相关信息”，你应统一回复：“关于这个问题，我当前的知识库中没有找到确切的资料。”

        # 回答流程（不可更改）：
        用户提问 -> 调用 `search_context` 工具 -> 基于工具返回结果生成答案
        。
        """)
    # system_message = SystemMessage(
    #     content='你是一个Flink分布式计算引擎的专家，如果输入给你的信息中包含相关内容，直接回答。否则不要自己直接回答用户的问题，一定调用工具和知识库来补充，生成最终的答案。')
    print(llm_with_tools.kwargs)
    return {"messages": [llm_with_tools.invoke([*state["messages"], system_message])]}
def second_chatbot(state:MultiModalRAGState):
    print(state['messages'])
    return {'messages':[LLM_QWEN.invoke(state['messages'])]}
def thrid_chatbot(state:MultiModalRAGState):
    context_retrive=state.get('context_retrive')
    image_retrive=state.get('image_retrive')

    count = 0
    context_pieces = []
    for hit in context_retrive:
        count+=1
        context_pieces.append(f"检索后的内容{count}：\n {hit.get('text')} \n 资料来源：{hit.get('filename')}")
    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息。"
    context = context.replace("{", "{{").replace("}", "}}")
    print(f'{context}+{image_retrive}')
    system_prompt = f"""
            请根据用户输入和以下检索到的上下文内容生成响应，如果上下文内容中没有相关答案，请直接说明，不要自己直接输出答案。
            要求：
            1. 响应必须使用Markdown格式
            2. 在响应文字下方显示所有相关图片，图片的路径列表为{image_retrive}，使用Markdown图片语法：
            3. 在相关图片下面的最后一行显示上下文引用来源（来源文件名）
            4. 如果用户还输入了图片，请也结合上下文内容，生成文本响应内容。
            5. 如果用户还输入了文本，请结合上下文内容，生成文本响应内容。

            上下文内容：
            {context}
            """
    input_text=state.get('inpute_context')
    input_image=state.get('inpute_image')
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_content)
        ]
    )

    chain = prompt | LLM_QWEN

    return {"messages": [chain.invoke({'context': context})]}
def human_approval(state: MultiModalRAGState):
    log.info('已经进入了人工审批节点')
    log.info(f'当前的状态中的人工审批信息：{state["user_answer"]}')


def fourth_chatbot(state: MultiModalRAGState):
    print('第四节点')
    search_count = state.get('websearch', 0)
    print(state['user_answer'])
    """网络搜索工具绑定的大模型，第四大模型调用"""
    llm_tools = LLM_QWEN.bind_tools(web_tools)
    if search_count > 3:
        zd_m = AIMessage(
            content="已经完成三次网络检索，请根据已有信息回答",
        )
        message = state['messages'][-6:]
        message.append(zd_m)
        return {'messages': [llm_tools.invoke(message)
                             ], 'websearch': 0}

    input_text = state.get('inpute_context')
    # result = interrupt({})  # 动态的人工介入， 中断后，人工开始填写人工回复。接下来开始恢复工作流（从开始节点到当前的interrupt点重复执行）
    system_message = SystemMessage(
        content='你是一个智能体助手，一定要优先调用互联网搜索工具，请根据用户输入和互联网搜索结果，生成回复。最多调用三次生成结果')
    message = HumanMessage(
        content=[
            {"type": "text", "text": input_text},
        ],
    )
    return {"messages": [llm_tools.invoke([system_message, message])], 'websearch': search_count + 1}

def build_graph(checkpointer=None, store=None):
    builder=StateGraph(MultiModalRAGState)
    # checkpoint.setup()
    # memory.setup()
    builder.add_node("process_input", process_inpute)

    builder.add_node("first_chatbot", frist_chatbot)
    # serch_node=SearchCOntextNode(tool)
    builder.add_node('search_context',ToolNode(tools=tool))
    builder.add_node("retriever_node", retriever_node)
    builder.add_node("second_chatbot", second_chatbot)
    builder.add_node("third_chatbot", thrid_chatbot)
    builder.add_node("evaluate_node",answer_envaluate)
    builder.add_node("human_approval", human_approval)
    builder.add_node("fourth_chatbot", fourth_chatbot)
    builder.add_node("web_search_node", ToolNode(tools=web_tools))
    builder.add_edge(START, 'process_input')
    builder.add_conditional_edges('process_input', route_only_image,
                                  {"retriever_node": "retriever_node", 'first_chatbot': 'first_chatbot'})

    builder.add_conditional_edges('first_chatbot', tools_condition, {"tools": "search_context", END: END}, )

    builder.add_conditional_edges('search_context', route_llm_or_retriever,
                                  {"retriever_node": "retriever_node", 'second_chatbot': 'second_chatbot'})

    builder.add_edge('retriever_node', 'third_chatbot')

    # builder.add_edge('second_chatbot', 'evaluate_node')  # 任何结果，都要进行RAGAS的评估

    builder.add_conditional_edges('third_chatbot', route_evaluate_node, {"evaluate_node": "evaluate_node", END: END}, )
    builder.add_conditional_edges('evaluate_node', route_human_node, {"human_approval": "human_approval", END: END}, )
    builder.add_conditional_edges('human_approval', route_human_approval_node,
                                  {"fourth_chatbot": "fourth_chatbot", END: END}, )
    builder.add_conditional_edges(
        "fourth_chatbot",
        tools_condition,
        {"tools": "web_search_node", END: END},
    )
    builder.add_edge('web_search_node', 'fourth_chatbot')
    return builder

    # graph = builder.compile(
    #     checkpointer=checkpoint,
    #     store=memory,
    #     interrupt_before=['human_approval']  # 添加中断点   静态的人工介入， 当恢复工作流时，会从中断点开始恢复工作流
    # )
    # draw_graph(graph, 'graph_rag.png')
    #
session_id = str(uuid.uuid4())
config = {
    "configurable": {
        "user_name": "ZS",
        # 检查点由session_id访问
        "thread_id": session_id,
    }
}
def active_graph():
    S_M.connect()
    checkpoint, memory= S_M.get_memory()
    print(checkpoint)
    graphs=build_graph()
    graph=graphs.compile(
                checkpointer=checkpoint,
                store=memory,
                interrupt_before=['human_approval']  # 添加中断点   静态的人工介入， 当恢复工作流时，会从中断点开始恢复工作流
            )
    return graph
def update_state(user_inpute:str, config):
    graph=active_graph()
    if user_inpute=='approve':
        new_message='approve'
    else:
        new_message='reject'
    graph.update_state(config=config,
                       values={
                           'user_answer':new_message
                       })
def transcribe_image(image_path):
            """
            将任意格式的图片转换为base64编码的data URL
            :param image_path: 图片路径
            :return: 包含base64编码的字典
            """
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_to_base64(image_path)[0]}",
                    "detail": 'low'
                }
            }

def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置,并返回后面的所有user消息"""
    if not history:
        return None
    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后查找第一个user
        return history[last_assistant_idx + 1:]

def add_message(history, user_input):
    """将用户输入的消息添加到聊天记录中"""
    if user_input['text'] is not None:  # 文本消息
        history.append({'role': 'user', "content": user_input['text']})

    for m in user_input['files']:
        print(m)
        history.append({'role': 'user', "content": {'path': m}})

    # 返回更新后的聊天历史记录和一个清空且不可交互的输入框
    return history, gr.MultimodalTextbox(value=None, interactive=False)

async def submit_llm(history: List[Dict]):
    graph=active_graph()

    user_input_messages = get_last_user_after_assistant(history)

    log.info(user_input_messages)
    content = []  # 把整个用户输入的消息暂存
    print(user_input_messages)
    if user_input_messages:
        for x in user_input_messages:

            if isinstance(x['content'], str):  # 文本消息
                content.append({'type': 'text', 'text': x['content']})
            elif isinstance(x['content'], tuple):  # 多媒体输入消息
                file_path = x['content'][0]  # 得到多媒体的文件路径
                if file_path.endswith(".jpg") or file_path.endswith(".png") or file_path.endswith(".jpeg"):
                    file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                pass
    input_message = HumanMessage(content=content)
    print(input_message)
    full_response = ""  # 累积完整响应
    # tool_calls = []  # 记录工具调用
    inputs = {'messages': [input_message]}
    current_state = graph.get_state(config)
    print(current_state)
    if current_state.next:
        user_input = history[-1]['content']
        update_state(user_input, config)
        inputs = None
    print(content)
    for chunk in graph.stream(
            inputs,
            config,
            stream_mode=['messages',"values"],  # 同时监听消息和状态更新
    ):
        print(chunk)
        if isinstance(chunk, tuple) and 'messages' in chunk:
            for message in chunk[1]:
                print(message)

                # 处理 AI 文本流式片段
                if isinstance(message, AIMessageChunk) and message.content:
                    full_response+= message.content

                    # 关键：创建新的 history 列表，不要修改原对象属性
                    new_history= list(history)

                # 检查最后一条是否是正在流式输出的 assistant 消息
                    if (new_history and
                                isinstance(new_history[-1], ChatMessage) and
                                new_history
                                [-1].role == "assistant" and
                                'title' not in new_history[-1].metadata):
                            # 关键：创建新的 ChatMessage，而不是修改 .content
                            last_meta= new_history[-1].metadata
                            new_history[-1] = ChatMessage(role ="assistant",content=full_response,metadata= last_meta)
                    else:
                        new_history.append(ChatMessage(role="assistant",
                        content=message.content))
                    print(f'AAAI:{new_history}')

                    # 关键：yield 新列表，并让出时间片给 Gradio 渲染
                    yield
                    new_history
                    await asyncio.sleep(0.01)  # Windows 下这个值不能太小，0.01~0.03 都可以
                    history= new_history  # 同步更新引用，供下次循环使用

                    print(message)

                    # 处理工具调用
                if isinstance(message, AIMessage) and message.tool_calls:
                        tool_msg= f"🔧 调用工具: {message.tool_calls[0]['name']}\n{message.tool_calls[0]['args']}"

                        new_history= list(history)
                        new_history.append(ChatMessage(
                            role="assistant",
                            content=tool_msg,
                            metadata={"title": f"🛠️ Used tool {message.tool_calls[0]['name']}"}
                        ))
                        print(f'TI:{new_history}')
                yield new_history
                await asyncio.sleep(0.01)
                history=new_history
        # print(chunk)
        # if isinstance(chunk, tuple) and 'messages' in chunk:
        #     for message in chunk[1]:
        #         if isinstance(message, AIMessageChunk) and message.content:
        #             full_response += message.content
        #             if history and isinstance(history[-1], ChatMessage) and 'title' not in history[
        #                 -1].metadata:
        #                 history[-1].content = full_response
        #             else:
        #                 history.append(ChatMessage(role="assistant", content=message.content))
        #                 print(f'AAAI:{history}')
        #             yield history
        #         print(message)
        #         if isinstance(message, AIMessage) and message.tool_calls:
        #             tool_msg = f"🔧 调用工具: {message.tool_calls[0]["name"]}\n{message.tool_calls[0]["args"]}"
        #             history.append(ChatMessage(role="assistant", content=tool_msg,
        #                                        metadata={"title": f"🛠️ Used tool {message.tool_calls[0]["name"]}"}))
        #             print(f'TI:{history}')
        #
        #             yield history
        current_state = graph.get_state(config)
    if current_state.next:
        output = ("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
                  "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！")
        # history.append(ChatMessage(role="assistant", content=output))
        # print(f'IIII:{history}')
        # yield history
        new_history = list(history)
        new_history.append(ChatMessage(role="assistant", content=output))
        print(f'IIII:{new_history}')
        yield new_history
        history = new_history
    else:
        # 异步写入响应到Milvus（把当前工作流执行后的最终结果，保存到上下文的向量数据库中）
        mess = current_state.values.get('messages', [])
        if mess:
            if isinstance(mess[-1], AIMessage):
                log.info(f"开始写入Milvus:")
                # 异步写入Milvus
                task = asyncio.create_task(  # 创建一个多线程异步任务
                    get_milvus_writer().async_insert(
                        context_text=mess[-1].content,
                        user=current_state.values.get('user', 'ZS'),
                        message_type="AIMessage"
                    )
                )

css = '''
#bgc {background-color: #7FFFD4}
.feedback textarea {font-size: 24px !important}
.message { font-family: monospace; }
'''

with gr.Blocks(title='多模态RAG项目', theme=gr.themes.Soft(), css=css) as instance:
    gr.Label('多模态RAG项目', container=False)

    chatbot = gr.Chatbot(type='messages', height=450, label='AI助手',
                         bubble_full_width=False,
                         render_markdown=True,  # 启用Markdown渲染
                         line_breaks=False  # 禁用自动换行符
                         )  # 聊天记录组件

    # 多模态输入框
    chat_input = gr.MultimodalTextbox(
        file_types=['png'],
        file_count='multiple',
        placeholder='请输入文字或者图片信息...',
        show_label=False,
        sources=['upload'],
    )

    chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input]
    ).then(
        submit_llm,
        [chatbot],
        [chatbot],
    ).then(  # 回复完成后重新激活输入框
        lambda: gr.MultimodalTextbox(interactive=True),  # 匿名函数重置输入框
        None,  # 无输入
        [chat_input]  # 输出到输入框
    )



# async def excute_grapgh(user_inpute:str)->str:
#     graph = await active_graph()
#     result=''
#     State=await graph.aget_state(config)
#     if State.next:
#         await update_state(user_inpute,config)
#         async for chunk in graph.astream(None, config, stream_mode='values'):
#             pretty_print_messages(chunk, last_message=True)
#     else:
#         image_base64 = None
#         text = None
#         if '&'in user_inpute:
#             text = user_inpute.split('&')[0]
#             image= user_inpute.split('&')[1]
#             if image and os.path.isfile(image):
#                 image_base64 = {
#                     "type": "image_url",
#                     "image_url": {"url": image_to_base64(image)[0]},
#                 }
#         elif os.path.isfile(user_inpute):
#             mage_base64 = {
#                 "type": "image_url",
#                 "image_url": {"url": image_to_base64(user_inpute)[0]},
#             }
#         else:
#             text = user_inpute
#         message = HumanMessage(
#             content=[
#             ]
#         )
#         if text:
#             message.content.append({'type': 'text', 'text': text})
#         if image_base64:
#             message.content.append(image_base64)
#         async for chunk in graph.astream({'messages': [message]}, config, stream_mode='values'):
#             pretty_print_messages(chunk, last_message=True)
#     current_state=await graph.aget_state(config)
#     if current_state.next:
#         result=("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
#               "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！")
#         return result
#
#     else:
#         amessage= await graph.aget_state(config)
#         messages=amessage.values.get('messages',[])
#         print(messages)
#         if messages:
#             if isinstance(messages[-1],AIMessage):
#                 log.info(f"开始写入Milvus:")
#                 task=asyncio.create_task(
#                     get_milvus_writer().async_insert(
#                         context_text=messages[-1].content,
#                         user=current_state.values.get('user', 'ZS'),
#                         message_type="AIMessage"
#                     )
#                 )
#                 await task
#     return result
# async def main():
#     while True:
#         user_input = input('用户输入(文本和图片用&隔开)：')
#         if user_input.lower() in ['exit', 'quit', '退出']:
#                 break
#
#         res = await excute_grapgh(user_input)
#         if res:
#             print('AI: ', res)
#     S_M.disconnect()


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # 关闭 gradio 导入过程中可能已经创建的 ProactorEventLoop
        try:
            old_loop = asyncio.get_event_loop()
            if not old_loop.is_running():
                old_loop.close()
        except RuntimeError:
            pass

        # 手动创建 SelectorEventLoop 并设为当前循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    instance.launch(debug=True)



















