from langgraph.constants import END

from workflow.state import MultiModalRAGState


def route_only_image(state: MultiModalRAGState):
    if state.get('input_typr')=='only_image':
        return 'retriever_node'
    else:
        return 'first_chatbot'

def route_llm_or_retriever(state: MultiModalRAGState):
    if messages:=state.get('messages',[]):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")
    if message.content or message.content == "没有找到相关的历史上下文信息":
        return "retriever_node"
    return 'second_chatbot'
def route_evaluate_node(state: MultiModalRAGState):
    if state.get('input_typr')=='only_image':
        return END
    return 'evaluate_node'
def route_human_node(state: MultiModalRAGState):
    """
    动态路由函数，如果评估后的分值低于0.7，则进入人工介入节点
    """

    if state.get('evaluate_score') >= 0.7:
        return END
    return 'human_approval'


def route_human_approval_node(state: MultiModalRAGState):
    """
    动态路由函数，如果用户输入的是：approve 则结束，否则进入网络搜索
    """

    if state.get('user_answer') == 'approve':
        return END
    return 'fourth_chatbot'
