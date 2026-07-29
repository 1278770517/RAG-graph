import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, WeightedRanker
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from embeding import CustomQwen3Embeddings
from llm import zhipu_client, LLM_QWEN
from milvus_db.collections_operator import client, CONTEXT_COLLECTION_NAME
from utils.embeding_uiliti import call_dashscope_once
from utils.log_utils import log
from workflow.envaluate import RAGEvaluator

evaluator_llm = LangchainLLMWrapper(LLM_QWEN)
evaluator_embeddings = LangchainEmbeddingsWrapper(CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B"))

# 创建RAG评估器
rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embeddings)
@tool('search_context',parse_docstring=True)
async def asearch_context(query:str=None,user_name:str=None):
    """
        根据用户的输入，检索与查询相关的历史上下文信息，然后给出正确的回答。

        Args:
            query: (可选)用户刚刚输入的文本内容。
            user_name: (可选)当前的用户名。

        Returns:
            从历史上下文中检索到的结果。

        """
    # 构建文本输入数据
    input_data = [{'text': query}]
    ok, embedding, status, retry_after = call_dashscope_once(input_data)
    filter_expr = ''
    if user_name:
        filter_expr = f'user == "{user_name}"'
    parm={"metric_type": "IP", "params": {"nprobe": 10}}
    dense_req=AnnSearchRequest(data=[embedding],anns_field='context_dense',param=parm
                     ,expr=filter_expr,limit=3)
    sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
    spare_req=AnnSearchRequest(
        [query], "context_sparse", sparse_search_params, limit=3, expr=filter_expr)
    weight=WeightedRanker(1.0,1.0)
    res=client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs=[spare_req, dense_req],
        ranker=weight,  # 重排算法
        limit=3,
        output_fields=["context_text"]
    )
    log.info(f'上下文检索结果：{res[0]}')
    context_pieces = []
    for hit in res[0]:
        context_pieces.append(f"{hit.get('context_text')}")
    score=await rag_evaluator.evaluate_context(query, context_pieces)
    log.info(f"上下文检索后，评估分数为: {score}")
    if score < 1.0:  # 评估分数小于1.0，则返回空
        context_pieces = []
    ''' 修改的代码 结束'''

    return "\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息。"


@tool('search_context',parse_docstring=True)
def search_context(query:str=None,user_name:str=None):
    """
        根据用户的输入，检索与查询相关的历史上下文信息，然后给出正确的回答。

        Args:
            query: (可选)用户刚刚输入的文本内容。
            user_name: (可选)当前的用户名。

        Returns:
            从历史上下文中检索到的结果。

        """
    # 构建文本输入数据
    input_data = [{'text': query}]
    ok, embedding, status, retry_after = call_dashscope_once(input_data)
    filter_expr = ''
    if user_name:
        filter_expr = f'user == "{user_name}"'
    parm={"metric_type": "IP", "params": {"nprobe": 10}}
    dense_req=AnnSearchRequest(data=[embedding],anns_field='context_dense',param=parm
                     ,expr=filter_expr,limit=3)
    sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
    spare_req=AnnSearchRequest(
        [query], "context_sparse", sparse_search_params, limit=3, expr=filter_expr)
    weight=WeightedRanker(1.0,1.0)
    res=client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs=[spare_req, dense_req],
        ranker=weight,  # 重排算法
        limit=3,
        output_fields=["context_text"]
    )
    log.info(f'上下文检索结果：{res[0]}')
    context_pieces = []
    for hit in res[0]:
        context_pieces.append(f"{hit.get('context_text')}")
    score=asyncio.run(rag_evaluator.evaluate_context(query, context_pieces))
    log.info(f"上下文检索后，评估分数为: {score}")
    if score < 1.0:  # 评估分数小于1.0，则返回空
        context_pieces = []
    ''' 修改的代码 结束'''

    return "\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息。"

class SearchInput(BaseModel):
    quer:str=Field(description='用户需要输入的问题')

@tool('web_search',args_schema=SearchInput,description='专门网页搜索的工具')
def web_search(quer:str=None):
    try:
        response=zhipu_client.web_search.web_search(search_query=quer,search_engine='search_std')
        if zhipu_client:
            return "\n\n".join([d.content for d in response.search_result])
        return '没有搜索到任何内容！'


    except Exception as e:
        print(e)
        return '没有搜索到结果'


