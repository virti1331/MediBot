import os
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        top_p=0.95,
        max_new_tokens=1024  
    )
    return llm

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("💊 MediBot - Your Medical Assistant")

    uploaded_file = st.file_uploader("📄 Upload a medical report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    extracted_text = ""

    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif "image" in uploaded_file.type:
                extracted_text = extract_text_from_image(uploaded_file)

            st.success("✅ Report text extracted successfully!")

            with st.expander("📝 Extracted Report Text"):
                st.markdown(f"```\n{extracted_text[:1000]}\n```")  
            if st.button("🧾 Summarize My Report"):
                HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                HF_TOKEN = os.environ.get("HF_TOKEN")

                summary_prompt = f"""
                You are a medical assistant. Read the following medical report text and generate a comprehensive summary.
                Focus on the patient’s condition, key lab results, diagnosis, and suggestions (if any).
                Make the summary helpful and medically meaningful.

                Report:
                {extracted_text}

                Summary:
                """

                llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
                summary = llm.invoke(summary_prompt)
                st.subheader("📋 Report Summary")
                st.markdown(summary)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a question about your report or anything medical...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say "I don't know". Don't make anything up.
        Only use the context provided below.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            context_source = extracted_text if extracted_text else ""
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            inputs = {"query": prompt, "context": context_source}
            response = qa_chain.invoke(inputs)
            result = response["result"]
            source_documents = response.get("source_documents", [])

            st.chat_message('assistant').markdown(result)

            if source_documents:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(source_documents):
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"**[{i + 1}] Source:** `{source}`")
                        st.markdown(f"```\n{doc.page_content[:500]}\n```")

            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
