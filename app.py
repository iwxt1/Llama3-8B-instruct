import gradio as gr
import os
import torch
from lmdeploy import pipeline, TurbomindEngineConfig


# download internlm2 to the base_path directory using git tool
base_path = './llama3-8b-chat'
os.system(f'git clone https://code.openxlab.org.cn/eoeterang/HealthGuardian.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="Llama3-Chat-8B-Medical",
                description="""
Llam3-Chat-8B-Medical is mainly developed to enhance medical QA quality.  
                 """,
                 ).queue(1).launch()