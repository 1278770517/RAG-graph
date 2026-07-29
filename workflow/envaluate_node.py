import asyncio

from langchain_core.messages import AIMessage
from ragas.embeddings import OpenAIEmbeddings
# from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper


from embeding import CustomQwen3Embeddings
from llm import LLM_QWEN
from utils.log_utils import log
from workflow.envaluate import RAGEvaluator
from workflow.state import MultiModalRAGState
from workflow.tools import evaluator_llm, evaluator_embeddings

envaluate_llm=LangchainLLMWrapper(LLM_QWEN)
envaluate_embedinng=OpenAIEmbeddings(CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B"))
rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embeddings)

def answer_envaluate(state:MultiModalRAGState):
    user_inpute=state.get('inpute_context')
    context_retrive=state.get('context_retrive')
    message=state['messages'][-1]
    if isinstance(message,AIMessage):
        answer=message.content
    score=asyncio.run(rag_evaluator.evaluate_answer(user_inpute,context_retrive,answer))
    log.info(f"RAG Evaluation Score: {score}")
    score = float(score)
    return {'evaluate_score':score}
