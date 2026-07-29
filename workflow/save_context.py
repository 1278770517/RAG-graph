import asyncio
from concurrent.futures import ThreadPoolExecutor
# from datetime import time
import time
from typing import Dict, Any

from pymilvus import MilvusClient

from embeding import CustomQwen3Embeddings
from milvus_db.collections_operator import client, CONTEXT_COLLECTION_NAME
from utils.log_utils import log

thread_pool=ThreadPoolExecutor(max_workers=5)

class OptimizedMilvusAsyncWriter:
    def __init__(self,client:MilvusClient,collection_name:str=="t_context_collection"):
        self.client = client
        self.collection_name = collection_name
        self.embeding=CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")

    def get_dence(self,query:str):
        try:
            dence_vactor=self.embeding.embed_query(query)
            return dence_vactor
        except Exception as e:
            log.exception(f"向量生成失败: {e}")
            return None
    def _sync_insert(self, data: Dict[str, Any]):
        try:
            result=self.client.insert(collection_name=self.collection_name, data=data)
            log.info(f"插入成功: {data}")
        except Exception as e:
            log.exception(f"插入数据到Milvus失败: {e}")
    async def async_insert(self, context_text: str, user: str, message_type: str = "AIMessage"):
        data = {
            "context_text": context_text,
            "user": user,
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "message_type": message_type,
            "context_dense": self.get_dence(context_text)
        }
        log.info(f"准备写入: {data}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool,self._sync_insert, data)
_milvus_writer_instance = None
def get_milvus_writer() -> OptimizedMilvusAsyncWriter:
    """获取全局Milvus写入器实例（单例）"""
    global _milvus_writer_instance
    if _milvus_writer_instance is None:
        _milvus_writer_instance = OptimizedMilvusAsyncWriter(
            client=client,
            collection_name=CONTEXT_COLLECTION_NAME
        )
    return _milvus_writer_instance



