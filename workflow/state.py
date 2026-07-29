from typing import Optional, List, Dict, Literal

from langgraph.graph import MessagesState


class MultiModalRAGState(MessagesState):
    # input_typr:输入数据类型
    # context_retrive:向量化文本
    # image_retrive:图片向量化
    input_type: Literal["has_text", "only_image"]
    context_retrive: Optional[List[Dict[str, str]]]
    image_retrive: Optional[List[str]]

    # need_retrive: 是否需要向量化
    # evaluate_retrive: 向量化评估
    # final_response:最终回复
    need_retrive: Optional[bool]
    evaluate_score: Optional[float]
    final_response: Optional[bool]

    # user_answer: 中断的用户回答
    # inpute_context: 用户输入文本
    # inputr_image: 输入图片
    # user_id: 用户id
    user_answer: Optional[str] = 'rejected'
    inpute_context: Optional[str]
    inpute_image: Optional[str]
    user_id: str = 'ZS'

    websearch:int


