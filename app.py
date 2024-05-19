import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


# download internlm2 to the base_path directory using git tool
base_path = './llama3-8b-chat-4bit'
os.system(f'git clone https://code.openxlab.org.cn/eoeterang/HealthGuardian_4bit.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="Llama3-Chat-8B-Medical-4bit",
                description="""
Llam3-Chat-8B-Medical is mainly developed to enhance medical QA quality.  
                 """,
                 ).queue(1).launch()

import os

os.system('streamlit run web_demo-Llama3.py --server.address=0.0.0.0 --server.port 7860')

model = "EmoLLM_aiwei"
# model = "EmoLLM_Model"
# model = "Llama3_Model"

if model == "EmoLLM_aiwei":
    os.system("python download_model.py ajupyter/EmoLLM_aiwei")
    os.system('streamlit run web_demo-aiwei.py --server.address=0.0.0.0 --server.port 7860')
elif model == "EmoLLM_Model":
    os.system("python download_model.py jujimeizuo/EmoLLM_Model")
    os.system('streamlit run web_internlm2.py --server.address=0.0.0.0 --server.port 7860')
elif model == "Llama3_Model":
    os.system("python download_model.py chg0901/EmoLLM-Llama3-8B-Instruct3.0")
    os.system('streamlit run web_demo_Llama3.py --server.address=0.0.0.0 --server.port 7860')
else:
    print("Please select one model")
