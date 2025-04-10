import os

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variable and set API token explicitly
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN  # ✅ explicitly set

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Setup LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # ✅ this fixes the task error
        temperature=0.5,
        model_kwargs={
            "max_new_tokens": 512,
            "top_p": 0.95
        }
    )
    return llm

# Step 2: Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say "I don't know." Don't make anything up.
Only use the context provided below.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Accept and run query
user_query = input("🧠 Write your medical question: ")
response = qa_chain.invoke({'query': user_query})

# Step 6: Print answer and sources
print("\n✅ RESULT:")
print(response["result"])

print("\n📚 SOURCE DOCUMENTS:")
for i, doc in enumerate(response["source_documents"]):
    print(f"\n[{i + 1}] Source: {doc.metadata.get('source', 'Unknown')}")
    print(doc.page_content[:500])
