import asyncio

from langchain_core.messages import ToolMessage

from milvus_db.collections_operator import client, COLLECTION_NAME
from milvus_db.db_retriever import MilvusRetriever
from utils.embeding_uiliti import call_dashscope_once
from utils.log_utils import log
from workflow.state import MultiModalRAGState

m_re=MilvusRetriever(milvus_client=client,collection_name=COLLECTION_NAME)


class SearchCOntextNode:
    def __init__(self,tools:list):
        self.tool_dict={tool.name:tool for tool in tools}
    async def __call__(self,tool_call:MultiModalRAGState):
        print(tool_call.keys())
        if messages := tool_call.get("messages", None):
            message=messages[-1]
        else:
            raise ValueError("No message found in input")
        use_tolls=[]
        for tool in message.tool_calls:
            if tool.get('args') and 'query' in tool['args']:
                query=tool['args']['query']
                log.info(f"query {query}")
            else:
                query=tool_call.get('inpute_context')
            use_toll=self.tool_dict[tool['name']].ainvoke({'query': query, 'user_name': tool_call['user_id']})
            use_tolls.append((tool,use_toll))
        out=[]
        tool_res=await asyncio.gather(*[task for _, task in use_tolls], return_exceptions=True)
        for (toll_call,_),tool_result in zip(use_tolls,tool_res):
            if isinstance(tool_result,Exception):
                tool_result = f"工具执行错误: {str(tool_result)}"
            out.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=toll_call['id'],
                        name=toll_call['name']
                    )
                )

        return {"messages": out}


def retriever_node(state: MultiModalRAGState):
    if state.get('input_type')=='only_image':
        image_impute=state.get('inpute_image')
        inpute_data=[{'image':image_impute}]
        ok, embedding, status, retry_after =call_dashscope_once(inpute_data)
        res=m_re.dense_search(embedding,limit=3)
    else:
        res=m_re.hybrid_search(call_dashscope_once([{'text':state.get('inpute_context')}])[1],state.get('inpute_context'),limit=3)
    images=[]
    out=[]
    for hit in res:
        if hit.get('category')== 'image':
            images.append(hit.get('image_path'))
        out.append({"text": hit.get("text"), "category": hit.get("category"), "image_path": hit.get("image_path"),
                     "filename": hit.get("filename"), })
    return {'context_retrive': out, 'image_retrive': images}







