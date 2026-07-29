import asyncio
import hashlib
from typing import List
import re
import unicodedata
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

from llm import LLM_QWEN
from load_wikipedia_datas import loaded_docs


class CleanMarkdownLLM(BaseChatModel):
    """包装任意 ChatModel，自动去除返回内容中的 Markdown 代码块"""

    def __init__(self, llm: BaseChatModel, **kwargs):
        self._llm = llm
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return f"clean_{self._llm._llm_type}"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self._llm._generate(messages, stop, run_manager, **kwargs)
        return self._clean(result)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await self._llm._agenerate(messages, stop, run_manager, **kwargs)
        return self._clean(result)

    def _clean(self, result: ChatResult) -> ChatResult:
        for gen in result.generations:
            if gen.message and gen.message.content:
                content = gen.message.content.strip()
                # 去除 ```json 和 ``` 包裹
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                gen.message.content = content.strip()
        return result
clean_llm = CleanMarkdownLLM(llm=LLM_QWEN)
data=Neo4jGraph(url='bolt://192.168.109.129:7687',
                username="neo4j",
                password="wcr123456",
                database="neo4j",)
data.query("""
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:`__Entity__`) REQUIRE e.id IS UNIQUE
""")
def connect_id(meta: dict,) :
    meta_content=meta.get('source') or meta.get('summary')+'|'+meta.get('title')
    return hashlib.md5(meta_content.encode('utf-8')).hexdigest()
def attach_doc_ids(docs: List[Document])->List[Document]:
    for i in docs:
        i.metadata=dict(i.metadata)
        i.metadata['id']=connect_id(i.metadata)
        i.metadata.setdefault('source', 'wikipedia')
        i.metadata.setdefault('title', i.metadata.get("title",""))
        i.metadata.setdefault('source', i.metadata.get("source",""))
    return docs
def nomerlize(s:str) -> str:
    if not isinstance(s, str):
        return s
    s=unicodedata.normalize('NFKD', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s



allowed_nodes = [
    "Person", "Organization", "Company", "Product", "Event",
    "Country", "City", "University", "Award", "Position"
]
# 关系名建议用大写蛇形，便于统一（示例覆盖“马云/阿里巴巴”常见关系）
allowed_relationships = [
    ("Person", "FOUNDER_OF", "Organization"),
    ("Person", "FOUNDER_OF", "Company"),
    ("Person", "CEO_OF", "Company"),
    ("Person", "CHAIRMAN_OF", "Company"),
    ("Person", "BORN_IN", "City"),
    ("Person", "CITIZEN_OF", "Country"),
    ("Person", "EDUCATED_AT", "University"),
    ("Organization", "HEADQUARTERED_IN", "City"),
    ("Company", "HEADQUARTERED_IN", "City"),
    ("Organization", "PART_OF", "Organization"),
    ("Company", "OWNS", "Company"),
    ("Company", "ACQUIRED", "Company"),
    ("Person", "AWARDED", "Award"),
    ("Person", "FOUNDED", "Company"),       # 兼容部分模型输出
    ("Organization", "FOUNDED_BY", "Person"),
    ("Company", "FOUNDED_BY", "Person"),
    ("Organization", "PARTNERED_WITH", "Organization"),
]
custom_prompt = custom_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledge graph extraction specialist. Extract entities and relationships from the text.

STRICT OUTPUT FORMAT - You must output a single JSON object with exactly two keys: "nodes" and "relationships".

**nodes** must be an array of objects. Each object MUST have:
- "id": string (the entity name)
- "type": string (entity type, e.g., Person, Company)
- "properties": array of objects. Each object MUST have:
  - "key": string
  - "value": string
  Example: "properties": [{{"key": "birth_date", "value": "1971"}}]

**relationships** must be an array of objects. Each object MUST have:
- "source_node_id": string (must match a node id exactly)
- "source_node_type": string
- "target_node_id": string (must match a node id exactly)
- "target_node_type": string
- "type": string (relationship type, e.g., FOUNDER_OF)
- "properties": array of objects with "key" and "value", same as nodes above.

FORBIDDEN: Do NOT use "label", "startNode", "endNode", "source", "target".
FORBIDDEN: Do NOT put properties as a plain string or object. It MUST be an array of {{key, value}} pairs.
You must return valid JSON only. 
Do not wrap the response in markdown code blocks (no ```json).
Return raw JSON string directly.
"""
    ),
    ("human", "{input}")
])

# 也可以限定要抽取的一些关键属性（能抽到更稳定的主键/查询特征）
node_properties = ["alias", "dob", "founded_year", "citizenship", "ticker", "hq_city", "hq_country"]
trans=LLMGraphTransformer(llm=LLM_QWEN,
                          allowed_nodes=allowed_nodes,
                          allowed_relationships=allowed_relationships,
                          node_properties=node_properties,
                          prompt=custom_prompt,
                         )
async def extract_graph(docs: List[Document]) -> List[GraphDocument]:
    print(docs)
    graph_document_list=await trans.aconvert_to_graph_documents(docs)
    print(graph_document_list)
    for i in graph_document_list:
        for j in i.nodes:
            j.id=nomerlize(j.id)
        for j in i.relationships:
            j.source.id=nomerlize(j.source.id)
            j.target.id=nomerlize(j.target.id)
    return graph_document_list
def upsert_to_neo4j(graph_documents):
    data.add_graph_documents(graph_documents,baseEntityLabel=True)

async def ingest_wikipedia_docs(docs: List[Document]):
    docs = attach_doc_ids(docs)
    gdocs = await extract_graph(docs)
    print(gdocs)
    upsert_to_neo4j(gdocs)
asyncio.run(ingest_wikipedia_docs(loaded_docs))

