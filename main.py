import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Load LLM (with fixed task support)
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",  # ✅ This fixes the ValueError
        temperature=0.5,
        max_new_tokens=256,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm


# Step 2: Define a custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know" and do not make anything up.
Only use the information in the context below.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Take query from user
query = input("Ask your question: ")
response = qa_chain.invoke({"query": query})

# Step 6: Display result
print("\n🔍 Answer:")
print(response["result"])
print("\n📚 Source Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"\n[{i + 1}] {doc.metadata.get('source', 'Unknown Source')}")
    print(doc.page_content[:500])  # Only show first 500 characters
