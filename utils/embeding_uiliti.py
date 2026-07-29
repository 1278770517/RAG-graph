import os.path
import time
from http import HTTPStatus
from typing import Tuple, List, Dict, Optional
import dashscope
import requests

from util import QWEN_API
from utils.log_utils import log

MAX_IMAGE_BYTES = 3 * 1024 * 1024
RPM_LIMIT = 120  # 每分钟最多调用次数（Requests Per Minute）
WINDOW_SECONDS = 60  # 限流时间窗口（秒），与RPM_LIMIT配合实现每分钟限流
class FixedWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.count=0
        self.current=time.monotonic()
    def acquire(self):
        now = time.monotonic()

        if now-self.current>self.window_seconds:
            self.current=now
            self.count=0
        if self.count>self.limit:
            passtime = now - self.current
            reminder_time=self.window_seconds - passtime
            print(f"[限速] 达到 {self.limit} 次请求，等待 {reminder_time:.2f}s...")
            time.sleep(passtime)
            self.count=0
            self.window_start = time.monotonic()

        self.count+=1
def image_to_base64(img: str) -> Tuple[str, str]:
    print(f'路径{img}')
    try:
        import mimetypes,base64
        images_type=mimetypes.guess_type(img)[0] or "image/png"
        with open(img, "rb") as f:
            images_base64 = base64.b64encode(f.read()).decode("utf-8")
        image_URI=f"data:{images_type};base64,{images_base64}"
        return image_URI,img
    except Exception as e:
        print(f"[图片] 本地文件转 base64 失败：{e}")
        log.exception(e)
        return "", ""
def normalize_image(img: str) -> Tuple[str, str]:
    raw=img.strip()
    low=img.lower()
    print(raw)
    try:
        if not img:
            return '',''
        if low.startswith("http://") or low.startswith("https://"):
            statue=requests.head(raw, timeout=5, allow_redirects=True)
            if statue.status_code==200:
                image_size=int(statue.header.get("content-length"))
                if image_size>MAX_IMAGE_BYTES:
                    print(f'{image_size} > {MAX_IMAGE_BYTES}')
                    return "", ""
            else:
                print(f'图片不可达')
                return "", ""
            return raw,raw
    except Exception as e:
        print(f"[图片] HEAD 检查异常：{e}")
        return "", ""
    print(raw)
    if os.path.isfile(raw):
        print('是文件图片')
        return image_to_base64(raw)


Limter=FixedWindowRateLimiter(RPM_LIMIT, WINDOW_SECONDS)
def call_dashscope_once(input_data: List[Dict]) -> Tuple[bool, List[float], Optional[int], Optional[float]]:
    Limter.acquire()
    print(input_data)
    # if input_data[0].get('text'):
    #     input_data[0]['text']=''.join(input_data[0].get('text'))
    #     print(input_data)
    #     inpute1={'contents':input_data}
    #     print(inpute1)
    try:
        embeding_model_qwen=dashscope.MultiModalEmbedding.call(api_key=QWEN_API,model='qwen3-vl-embedding'
                                                               ,input=input_data,dimension=1024)

    except Exception as e:
        print(f"调用 DashScope 异常：{e}")
        log.exception(e)
        return False, [], None, None
    re_try=0
    try:
        header=getattr(embeding_model_qwen, "header", '')
        if header and isinstance(header, dict):
            ra = header.get("Retry-After") or header.get("retry-after")
            if ra:
                re_try=float(ra)
    except Exception as e:
        print(e)
        pass
    try:
        status=getattr(embeding_model_qwen, "status_code", '')
        code=getattr(embeding_model_qwen,'code','')
        if status==HTTPStatus.OK:
            try:
                embeding_out=embeding_model_qwen['output']['embeddings'][0]['embedding']
                return True, embeding_out, status, re_try
            except Exception as e:
                print(e)
                return False, [], status, re_try
        else:
            print(f'{status} 访问失败')
            return False, [], status, re_try
    except Exception as e:
        print(e)
        return False, [], status, re_try
def process_item_with_guard(item: Dict) -> Dict:
    new_item=item.copy()

    content=(new_item.get('text') or '').strip()
    image=(new_item.get('image_path') or '').strip()

    if image:
        image_uri,image=normalize_image(image)
        inpute_data=[{"image": image_uri, "text": content}]
        print(inpute_data)
    else:
        inpute_data=[{"text": content}]
    ok,embeding_out,status,re_try=call_dashscope_once(inpute_data)
    print(ok)
    if ok:
        new_item['dense'] = embeding_out
    else:
        new_item['dense'] = []

    return new_item

if __name__=='__main__':
    dict={'text': ' ：Journal of\nMarine Science\nand Engineering  \nMDPI  \nArticle', 'category': 'text', 'filename': 'jmse-12-00046', 'filetype': '', 'image_path': r'D:\GraphRAG\out\0a92ff7d8372463af56bb57cbf8d9c76.png', 'title': ''}
    print(process_item_with_guard(dict))







