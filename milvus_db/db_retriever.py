import os.path
from typing import List, Dict

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from utils.embeding_uiliti import image_to_base64, call_dashscope_once


class MilvusRetriever:
    def __init__(self,collection_name: str, milvus_client: MilvusClient, top_k: int =3):
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.top_k = top_k
    def  dense_search(self, query_dense_embedding, limit=10):
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res=self.milvus_client.search(collection_name=self.collection_name,
            data=[query_dense_embedding],
            anns_field="dense",
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params,)
        return res[0]

    def sparse_search(self, query_sparse_embedding, limit=10):
        res=self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_sparse_embedding,
            anns_field='spares',
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}
        )[0]
        return res

    def hybrid_search(
            self,
            query_dense_embedding,
            query_sparse_embedding,
            sparse_weight=1.0,
            dense_weight=1.0,
            limit=10,
    ):
        dense_ann_search=AnnSearchRequest(data=[query_dense_embedding],
                                          anns_field="dense",
                                          limit=limit,
                                          param={"metric_type": "IP", "params": {"nprobe": 10}})
        sparse_ann_search=AnnSearchRequest(data=[query_sparse_embedding],
                                           anns_field="sparse",
                                           limit=limit,
                                           param={"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}})
        rerank = WeightedRanker(sparse_weight, dense_weight)
        return self.milvus_client.hybrid_search(collection_name=self.collection_name,
                                                reqs=[dense_ann_search,sparse_ann_search],
                                                ranker=rerank,
                                                limit=limit,
                                                output_fields=["text", 'category', 'filename', 'image_path', 'title'])[0]

    def retrieve(self, query)->List[Dict]:
        if os.path.isfile(query):
            image_data=[{'image': image_to_base64(query)[0]}]
            ok,embdedings,status,_=call_dashscope_once(image_data)
        else:
            text_data=[{'text': query}]
            ok,embdedings,status,_=call_dashscope_once(text_data)
        if ok:
            if os.path.isfile(query):
                res=self.dense_search(embdedings, limit=self.top_k)
            else:
                res=self.sparse_search(embdedings, limit=self.top_k)
        docs = []
        print(res)
        for hit in res:
            docs.append({"text": hit.get("text"), "category": hit.get("category"), "image_path": hit.get("image_path"), "filename": hit.get("filename"),})

        return docs


    def hybrid_search1(
            self,
            query_dense_embedding,
            query_sparse_embedding,
            sparse_weight=1.0,
            dense_weight=1.0,
            limit=10,
    ):
        dense_ann_search=AnnSearchRequest(data=[query_dense_embedding],
                                          anns_field="dense",
                                          limit=limit,
                                          param={"metric_type": "IP", "params": {"nprobe": 10}})
        sparse_ann_search=AnnSearchRequest(data=[query_sparse_embedding],
                                           anns_field="sparse",
                                           limit=limit,
                                           param={"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}})
        search1=[dense_ann_search,sparse_ann_search]
        search2=dense_ann_search
        rerank = WeightedRanker(sparse_weight, dense_weight)
        return self.milvus_client.hybrid_search(collection_name=self.collection_name,
                                                reqs=[search1,search2],
                                                ranker=rerank,
                                                limit=limit,
                                                output_fields=["text", 'category', 'filename', 'image_path', 'title'])[0]




