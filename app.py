import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        except Exception as e:
            st.error(f"Error reading {getattr(pdf, 'name', 'PDF')}: {e}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("No readable text found in the uploaded PDF(s).")
        return False
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return True


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all relevant details.
    If the answer is not in the provided context, just say "answer is not available in the context", do not provide incorrect information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ No processed documents found. Please upload PDF files and click 'Submit & Process' first.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.info("No matching content found for your query.")
            return

        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        output_text = response.get("output_text", "")
        st.write("### Reply:")
        st.write(output_text)
    except Exception as e:
        st.error(f"Error processing your query: {e}")


def main():
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon="📚")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    if not api_key:
        st.error("🔑 Google API Key is missing! Please configure `GOOGLE_API_KEY` in your `.env` file.")
        st.stop()

    user_question = st.text_input("Ask a question about your uploaded PDF documents: ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        if os.path.exists("img/robot_avatar.jpg"):
            st.image("img/robot_avatar.jpg")
        st.write("---")
        
        st.title("📁 PDF Files Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF files & click 'Submit & Process':",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("⚠️ Please select at least one PDF file before clicking submit.")
            else:
                with st.spinner("Extracting text and generating vector index..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Could not extract any text from the provided PDF(s). Please make sure they contain readable text.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        success = get_vector_store(text_chunks)
                        if success:
                            st.success("✅ Vector index built successfully! You can now ask questions.")
        
        st.write("---")
        st.write("AI App created by @ BHANU K")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 10px; text-align: center; font-size: 13px;">
            Multi-PDF Chatbot Powered by Gemini & LangChain
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
