import gradio as gr
import os
import torch
from lmdeploy import pipeline, TurbomindEngineConfig


# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1_8b-4bit'
os.system(f'git clone -b master https://code.openxlab.org.cn/eoeterang/internlm2_1_8b_4bit.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.4)

pipe = pipeline('./internlm2-chat-1_8b-4bit', backend_config=backend_config)


gr.ChatInterface(pipe,
                 inputs=[gr.Textbox()],
                 outputs=gr.Chatbot()
                 description="""
                    InternLM 1.8b chat 4bit
                 """,
                 ).launch()