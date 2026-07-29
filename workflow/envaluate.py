from typing import List, Dict

from ragas import SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings
# from ragas.embeddings.base import LangchainEmbeddingsWrapper
# from ragas.llms.base import LangchainLLMWrapper
from ragas.llms import llm_factory
from ragas.metrics import ContextRelevance
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference


from llm import LLM_QWEN
from milvus_db.collections_operator import client, COLLECTION_NAME
from milvus_db.db_retriever import MilvusRetriever
from utils.log_utils import log


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    使用LLM基于检索到的上下文生成文本答案

    Args:
        question: 用户问题 (中文)
        contexts: 检索到的上下文列表 (包含text、category等字段)

    Returns:
        生成的中文答案
    """
    # 将检索到的上下文格式化为字符串，便于LLM理解
    # 每个上下文前加上"上下文X"标识，方便LLM区分
    context_str = "\n\n".join([f"上下文 {i + 1}: {context['text']}" for i, context in enumerate(contexts)])

    # 提示词模板 (已翻译成中文)
    prompt = f"""
你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保你的回答基于提供的上下文，不要添加额外信息。

用户问题: {question}

检索到的上下文:
{context_str}

请基于以上上下文回答用户问题。
"""

    # 调用LLM生成答案
    response = LLM_QWEN.invoke(prompt)
    return response.content

class RAGEvaluator:
    def __init__(self, evaluator_llm:str,evaluator_embeddings:List[str]):
        self.evaluator_llm = evaluator_llm
        self.evaluator_embeddings=evaluator_embeddings
    async def evaluate_context(self, sample:str,retrieved_contexts:List[str]):
        """上下文相关性评估: 检索到的上下文（块或段落）是否与用户输入相关。"""
        # 0 → 检索到的上下文与用户查询完全不相关。
        # 1 → 上下文部分相关。
        # 2 → 上下文完全相关。
        # SingleTurnSample用于表示单轮对话的评估样本
        envaluate=SingleTurnSample(
            user_input=sample,
            retrieved_contexts=retrieved_contexts
        )

        out =ContextRelevance(llm=self.evaluator_llm)

        return await out.single_turn_ascore(envaluate)
    async def evaluate_answer(self,sample:str,retrived_contexts:List[Dict],respon:str):
        envaluate=SingleTurnSample(user_input=sample,
                                   retrieved_contexts=[content['text'] for content in retrived_contexts],
                                   response=respon)
        out=ResponseRelevancy(llm=self.evaluator_llm,embeddings=self.evaluator_embeddings)
        return await out.single_turn_ascore(envaluate)

    async def evaluate_metrics(self, sample:str,retrieved_contexts:List[Dict],response:str,reference:str=None):
        """评估生成的答案质量"""
        # SingleTurnSample用于表示单轮对话的评估样本
        envaluate=SingleTurnSample(user_input=sample,
                                   retrieved_contexts=[context['text'] for context in retrieved_contexts],
                                   response=response,
                                   reference=reference
                                   )
        log.info(f"开始评估答案质量, 评估样本为：{envaluate}")
        if reference:
            out=LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        else:
            out=LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        return await out.single_turn_ascore(envaluate)
async def RAG_evaluate(self,question:str):
    """
           评估RAG模型

           Args:
               question: 用户问题
               contexts: 检索到的上下文列表
               response: LLM生成的答案
               reference: 可选，参考答案 (用于评估的基准答案，通常为已知的正确答案)
           """
    # 1. 创建评估样本 (SingleTurnSample)
    envaluate_llm=llm_factory(self.evaluator_llm)
    envaluate_embedings=OpenAIEmbeddings(self.evaluator_embeddings)

    search=MilvusRetriever(milvus_client=client,collection_name=COLLECTION_NAME)
    search_content=search.retrieve(question)
    ini_envaluate=RAGEvaluator(evaluator_llm=envaluate_llm,evaluator_embeddings=envaluate_embedings)
    get_answer=generate_answer(question=question,contexts=search_content)
    return await ini_envaluate.evaluate_answer(question,search_content,get_answer)

