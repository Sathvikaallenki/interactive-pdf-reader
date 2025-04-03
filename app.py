import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import chromadb
import os
import fitz 

chromadb.api.client.SharedSystemClient.clear_system_cache() 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") 

st.set_page_config(page_title='Interactive Reader', page_icon="📚")
st.title('Interactive Reader📚') 

st.subheader("Your documents")
uploaded_file = st.file_uploader('Upload your PDF here and click on "Process"', type=['txt', 'pdf']) 

question = st.text_input("Ask a question from the PDF:",  placeholder = "Enter your question here") 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Check if the file is a PDF
        if uploaded_file.type == "application/pdf":
            document_text = read_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode()

        documents = [document_text]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        # Select embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Create a vectorstore from documents
        database = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = database.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        # Create QA chain
        response = rag_chain.invoke(query_text)
        return response

result = None
with st.form('chatbot', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Process', disabled=not(uploaded_file ))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, question)
            result = response

if result:
    # Chat container style
    st.markdown(
        """
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-top: 20px;
        }
        .chat-bubble {
            border-radius: 15px;
            padding: 20px;
            max-width: 100%;
        }
        .user {
            background-color: #2b3a50;
            color: white;
            align-self: flex-end;
        }
        .bot {
            background-color: #e8ebef;
            color: black;
            align-self: flex-start;
        }
        .avatar-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 5px;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
        }
        .username {
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Chat rendering section
    with st.expander("You Chat ", expanded=True):
        st.markdown(
        """
         <div style="margin-bottom: 10px;">
        <div class="avatar-row" style="justify-content: flex-end;">
            <span class="username">You</span>
            <img src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png" class="avatar">
        </div>
        <div class="chat-bubble user">
            {}</div>
            </div>
        """.format(question),
        unsafe_allow_html=True
    )

        st.markdown(
        """
        <div style="margin-top: 15px;">
        <div class="avatar-row">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar">
            <span class="username">Reader Bot</span>
        </div>
        <div class="chat-bubble bot">
            {}
        </div>
        </div>
        <div style="margin-top: 15px;"></div>
        """.format(result),
        unsafe_allow_html=True
    )
