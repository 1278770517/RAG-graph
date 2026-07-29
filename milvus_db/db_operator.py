import json
import os
import time
import random
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pymilvus import MilvusException

import mian
from llm import LLM_QWEN_true
from milvus_db.collections_operator import client, COLLECTION_NAME
from splister.splister_md import Markdownspliter
from utils.common_utils import get_surrounding_text_content
from utils.embeding_uiliti import image_to_base64, Limter, process_item_with_guard

RETRY_ON_429 = True  # 是否在遇到429（请求过多）状态码时进行重试
MAX_429_RETRIES = 5  # 429状态码的最大重试次数
BASE_BACKOFF = 2.0

def doc_to_dict(docs: List[Document]) -> List[Dict]:
    result = []
    for doc in docs:
        doc_dict={}
        metadata=doc.metadata
        if metadata.get('embedding_type') == 'text':
            doc_dict['text'] = doc.page_content
        else:
            doc_dict['text'] = None  # 或者设置为空字符串 ''，根据需要调整

        # 2. 提取 category (embedding_type)
        doc_dict['category'] = metadata.get('embedding_type', '')

        # 3. 提取 filename (source)
        source = metadata.get('source', '')
        doc_dict['filename'] = source

        # 4. 提取 filetype (source 中文件名的后缀)
        _, file_extension = os.path.splitext(source)
        doc_dict['filetype'] = file_extension.lower()  # 转换为小写，如 '.jpg'

        # 5. 提取 image_path (仅当 embedding_type 为 'image' 时)
        if metadata.get('embedding_type') == 'image':
            doc_dict['image_path'] = doc.page_content
        else:
            doc_dict['image_path'] = None  # 或者设置为空字符串 ''，根据需要调整
        header=[]
        header_list=[h for h in metadata.keys() if h.startswith('Header')]
        header_list1=sorted(header_list,key=lambda x:int(x.split()[1]) if x.split()[1].isdigit() else x)
        for key in header_list1:
            value = metadata.get(key, '').strip()
            header.append(value)
        doc_dict['title'] = "--> ".join(header) if header else ''
        if not doc_dict['image_path']:

            doc_dict['text'] = doc_dict['title'] + ' ：' + doc_dict['text']
        result.append(doc_dict)
    return result
def write_to_milvus(processed_data: List[Dict]):
    if not processed_data:
        print('无数据可以写入')
    else:
        try:
            client.insert(collection_name=COLLECTION_NAME,data=processed_data)
        except MilvusException as e:
            print(e)

def generate_image_description(data_list):
    context_prompt = ''
    for index,item in enumerate(data_list):
       print(item.get('image_path'))
       if item.get('image_path'):
           prev_text, next_text = get_surrounding_text_content(data_list, index)
           # print(prev_text, next_text)
           image_uri,image=image_to_base64(item['image_path'])
           if prev_text and next_text:
               context_prompt = f"""
                          前文内容: {prev_text}
                          后文内容: {next_text}

                          请根据以上上下文和图片内容，生成对该图片的简洁描述，描述内容长度最好不超过300个汉字。
                          注意：图片可能与前文、后文或两者都相关，请综合分析。
                          """
           elif prev_text:
               context_prompt = f"""
                          前文内容: {prev_text}

                          请根据以上上下文和图片内容，生成对该图片的简洁描述，描述内容长度最好不超过300个汉字。
                          注意：图片可能与前文内容相关，请结合分析。
                          """
           elif next_text:
               context_prompt = f"""
                          后文内容: {next_text}

                          请根据以上上下文和图片内容，生成对该图片的简洁描述，描述内容长度最好不超过300个汉字。
                          注意：图片可能与后文内容相关，请结合分析。
                          """
           else:
               context_prompt = "请描述这张图片的内容，生成对该图片的简洁描述，描述内容长度最好不超过300个汉字。"
           print(context_prompt)
               # 构建多模态消息
           message = HumanMessage(
               content=[
                   {"type": "text", "text": context_prompt},
                   {
                       "type": "image_url",
                       "image_url": {
                           "url": f"{image_uri}"
                       }
                   }
               ]
           )
           res=LLM_QWEN_true.invoke([message])
           item['text'] =res.content
    return data_list
def do_save_to_milvus(data: List[Document]):
    print(data)
    new_process_data=generate_image_description(doc_to_dict(data))
    data:List[Dict]=[]
    for index,item in enumerate(new_process_data):
        Limter.acquire()
        if RETRY_ON_429:
            attempt=0
            while True:
                result=process_item_with_guard(item.copy())
                if result.get('dense'):
                    data.append(result)
                    break
                attempt+=1
                if attempt>MAX_429_RETRIES:
                    print(f"[429重试] 超过最大重试次数，跳过 idx={index}")
                    data.append(result)
                    break
                backoff = BASE_BACKOFF * (2 ** (attempt - 1)) * (0.8 + random.random() * 0.4)
                print(f"[429重试] 第{attempt}次，sleep {backoff:.2f}s …")
                time.sleep(backoff)
        else:
            result = data.append(process_item_with_guard(item.copy()))
        if index % 20 == 0:
            print(f"[进度] 已处理 {index}/{len(new_process_data)}")

        # 打印处理后的 item 内容
    for item in data:
        print(json.dumps(item, ensure_ascii=False, indent=4))
    write_to_milvus(data)
    return data

if __name__ == '__main__':
    md_dir = r'D:\GraphRAG\output\jmse-12-00046'

    splitter = Markdownspliter(save_photo_dir=r'D:\GraphRAG\out')
    docs = splitter.process_md_dir(md_dir, source_filename='jmse-12-00046')
    # doc1=doc_to_dict(docs)
    # print(generate_image_description(doc1)[0])
    do_save_to_milvus(docs)


















