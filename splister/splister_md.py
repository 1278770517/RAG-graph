import base64
import hashlib
import io
import os
import re
from typing import List, Dict

from PIL import Image
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TextSplitter, MarkdownHeaderTextSplitter

from embeding import CustomQwen3Embeddings
from utils.common_utils import get_sorted_md_files


class Markdownspliter:
    def __init__(self,save_photo_dir:str,chunk_size:int=100):
        self.save_photo_dir=save_photo_dir
        self.chunk_size=chunk_size
        os.makedirs(self.save_photo_dir,exist_ok=True)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
        self.TextSpliter=MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        self.semantic_splitter=SemanticChunker(embeddings=self.embedding,number_of_chunks=self.chunk_size)

    def save_2_image(self,image_code:str,save_photo_dir:str):
       # print(image_code)
       if image_code.startswith("data:image"):
            image_code=image_code.split(",",1)[1]
       image=base64.b64decode(image_code)
       image=Image.open(io.BytesIO(image))
       image.save(save_photo_dir)

    def process_images(self, content: str, source: str) -> List[Document]:
        pattern=r'data:image/(.*?);base64,(.*?)\)'
        document_list=[]
        def replace_image(match):
            image_type = match.group(1).split(';')[0]
            base64_data = match.group(2)
            only_code=hashlib.md5(base64_data.encode()).hexdigest()
            filename = f"{only_code}.{image_type if image_type in ['png', 'jpg', 'jpeg'] else 'png'}"
            save_photo_dir=os.path.join(self.save_photo_dir,filename)
            self.save_2_image(base64_data,save_photo_dir)
            document_list.append(Document(
                page_content=str(save_photo_dir),
                metadata={
                    "source": source,
                    "alt_text": "图片",
                    "embedding_type": "image"
                }
            ))
            # print(document_list)
            return '[image]'
        conten=re.sub(pattern, replace_image, content, flags=re.DOTALL)
        print(document_list)
        return document_list
    def remove_pthoto(self,content:str):
        pattern = r'data:image/(.*?);base64,(.*?)\)'
        return re.sub(pattern, "", content)

    def process_md_file(self, md_file: str) -> List[Document]:
        with open(md_file,"r",encoding="utf-8") as f:
            content = f.read()
        document_list=[]
        split_txt=self.TextSpliter.split_text(content)
        for doc in split_txt:
            if '![](data:image/png;base64' in doc.page_content:
                image_doc: Document=self.process_images(doc.page_content,md_file)
                strip_pthoto=self.remove_pthoto(doc.page_content)
                if strip_pthoto.strip():
                    doc.metadata['embedding_type'] = 'text'
                    document_list.append(Document(page_content=strip_pthoto,metadata=doc.metadata))
                document_list.extend(image_doc)
            else:
                doc.metadata['embedding_type'] = 'text'
                document_list.append(doc)
        final_list=[]
        for doc in document_list:
            if len(doc.page_content)>self.chunk_size:
                final_list.extend(self.semantic_splitter.split_documents([doc]))
            else:
                final_list.append(doc)

        return final_list
    def add_title_hierarchy(self, documents: List[Document], source_filename: str) -> List[Document]:
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []
        for doc in documents:
            new_mandata=doc.metadata.copy()
            new_mandata['source']=source_filename
            for level in range(1,4):
                headre_level=f'Header {level}'
                if headre_level in new_mandata:
                    current_titles[level]=new_mandata[headre_level]
                    for level1 in range(level+1,4):
                        current_titles[level1]=''
            for level2 in range(1,4):
                headre_level = f'Header {level}'
                if headre_level not in new_mandata:
                    new_mandata[headre_level]=current_titles[level2]
                elif new_mandata[headre_level] != current_titles[level2]:
                    new_mandata[headre_level] = current_titles[level]
            processed_docs.append(Document(
                page_content=doc.page_content,
                metadata=new_mandata,
            ))
        return processed_docs

    def process_md_dir(self, md_dir: str, source_filename: str) -> List[Document]:
        file_list=get_sorted_md_files(md_dir)
        all_documents = []
        for file in file_list:
            all_documents.extend(self.process_md_file(file))
        return self.add_title_hierarchy(all_documents, source_filename)





if __name__ == '__main__':
    md_dir = r'D:\GraphRAG\output\jmse-12-00046'

    splitter = Markdownspliter(save_photo_dir=r'D:\GraphRAG\out')
    docs = splitter.process_md_dir(md_dir, source_filename='jmse-12-00046')

    # res: List[Dict] = do_save_to_milvus(docs)
    # 打印结果
    for i, doc in enumerate(docs):
        print(f"\n文档 #{i + 1}:")
        print(doc)








