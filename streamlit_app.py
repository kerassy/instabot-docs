import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

os.environ["GROQ_API_KEY"] = "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"

# Define embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

# Define language model
#llm = ChatGroq(temperature=1, model_name="mixtral-8x7b-32768")
llm = ChatGroq(model_name="gemma2-9b-it")


    
docs = PyMuPDFLoader(pdf_url).load()
    
# Split the PDF into chunks ready for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    length_function = len,
)

split_chunks = text_splitter.split_documents(docs)

# Create a vector database and temporary collection
qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="temp_pdf",
)
    
qdrant_retriever = qdrant_vectorstore.as_retriever()

def generate_response(input_text):
    # Define prompt
    RAG_PROMPT = """
    CONTEXT:
    {context}

    QUERY:
    {question}

    You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    rag_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | rag_prompt | llm | StrOutputParser()
    )
    response = rag_chain.invoke({"question" : input_text})
    return response

### Streamlit app
st.title("🗎 Instabot Docs")

pdf_url = st.sidebar.text_input("Link to pdf:", type="default", help="e.g. https://arxiv.org/pdf/2410.15608v2")
if st.sidebar.button("Read PDF"):
    create_vector_db()

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Can you give me a brief summary please?",
    )
    
    submitted = st.form_submit_button("Submit")
    
    if not pdf_url.startswith("htt"):
        st.warning("Please enter the direct URL of the .pdf document itself, e.g. https://arxiv.org/pdf/2410.15608v2 or https://www.site.com/document.pdf", icon="⚠")
    if submitted:
        answer = generate_response(text)
        st.write(answer)