import os

os.system('streamlit run web_demo-Llama3.py --server.address=0.0.0.0 --server.port 7860')

model = "HealthGuardian_4bit"

if model == "HealthGuardian_4bit":
    os.system("python download_model.py eoeterang/HealthGuardian_4bit")
    os.system('streamlit run web_demo-HG.py --server.address=0.0.0.0 --server.port 7860')
else:
    print("Please select one model")
