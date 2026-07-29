from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
from util import QWEN_URl, QWEN_API, XIAOAI_API, XIAOAI_URL

LLM_QWEN_true=ChatOpenAI(
    model='qwen3.5-omni-flash',
    base_url=QWEN_URl,
    api_key=QWEN_API,
)
# res=LLM_QWEN.invoke('你好')
# print(res)

print(XIAOAI_API)
LLM_QWEN=ChatOpenAI(
    model="qwen-max",
    openai_api_key=QWEN_API,
    base_url=QWEN_URl,
    # model_kwargs={
    #     "response_format": {"type": "json_object"},
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistant designed to output JSON."}
    #     ]
    # }
)
    # 可选
zhipu_client=ZhipuAI(api_key='178b205a2b1641c0a5fcab1b8f7be8ea.G5WPTByP4UUn313m')
