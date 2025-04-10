import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Load .env variables (optional if already set in system)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the LLM with correct parameters
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",  # Set task
    temperature=0.7,         # ✅ Moved outside model_kwargs
    max_new_tokens=256       # ✅ Moved outside model_kwargs
)

query = "What are the symptoms of diabetes?"
response = llm.invoke(query)

print("Query:", query)
print("Response:", response)
