import os

import gradio as gr
from huggingface_hub.cli import upload

from utils.common_utils import get_filename, delete_directory_if_non_empty, get_sorted_md_files
from utils.log_utils import log
from dots_ocr.parser import do_parse
base_md_dir = r'D:\GraphRAG\output'
class ProcessorAPP:
    def __init__(self):
        self.pdf_path = None
        self.md_dir= None
        self.md_file= None
        self.file_contents ={}
    def upload_pdf(self,pdf_file_path):
        log.info("Uploading PDF")
        self.pdf_path = pdf_file_path
        if self.pdf_path :
            return [gr.Button(interactive=True), f"PDF已上传: {os.path.basename(self.pdf_path)}"]
        else:
            return [gr.Button(interactive=False)]
    def parse_pdf(self):
        md1_dir=os.path.join(base_md_dir,get_filename(self.pdf_path, False))
        print(md1_dir)
        delete_directory_if_non_empty(md1_dir)
        do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True)
        if os.path.isdir(md1_dir):
            self.md_dir = md1_dir
            log.info(f"md文件路径{self.md_dir}")
            self.md_file = get_sorted_md_files(md1_dir)
            log.info(f"md_file{self.md_file}")
            for f in self.md_file:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        self.file_contents[f]=file.read()
                except Exception as e:
                    print(f"读取文件 {f} 时出错: {e}")
                    self.file_contents[f] = f"读取文件内容时出错: {e}"
            md_flie_list=[os.path.basename(f) for f in self.md_file]
            return [
                f"解析完成，共 {len(self.md_file)} 个MD文件",
                gr.Dropdown(label="MD文件列表",choices=md_flie_list,interactive=True),
                gr.Button(interactive=True),
                gr.Button(interactive=True)
            ]
        else:
            return [
                f"解析失败",
                gr.Dropdown(label="MD文件列表", interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False)]
    def select_md_file(self, selected_file):
        print(selected_file)
        print(self.file_contents.keys())
        if selected_file:
            for f in self.md_file:
                show_file = None
                if os.path.basename(f)==selected_file:
                    show_file=f
                    break
            if show_file and show_file in self.file_contents:
                return self.file_contents[show_file]
            else:
                return "没有找到该文件"
        else:
            return "文件内容加载失败,选择的文件不对"

    def create_interface(self):
        with gr.Blocks() as blocks:
            with gr.Row() as row:
                pdf_file_path=gr.File(label='选择pdf文件')
                upload_pdf=gr.Button(value='解析pdf', variant="primary", interactive=False)
            status=gr.Textbox(label='日志',value="等待操作", interactive=False)
            with gr.Row()as row1:
                Md_DropDown=gr.Dropdown(label='MD文件列表',interactive=False)
                Md_content=gr.Textbox(label='文件内容',value="等待展示", interactive=False)
            save_to_rag=gr.Button(value='存入知识库', variant="secondary", interactive=False)
            pdf_file_path.change(inputs=[pdf_file_path],outputs=[upload_pdf,status],fn=self.upload_pdf)
            upload_pdf.click(inputs=[],outputs=[status,Md_DropDown,upload_pdf,save_to_rag],fn=self.parse_pdf)
            Md_DropDown.change(inputs=[Md_DropDown],outputs=[Md_content],fn=self.select_md_file)
        return blocks
if __name__ == '__main__':
    c=ProcessorAPP()
    app=c.create_interface()
    app.launch()


