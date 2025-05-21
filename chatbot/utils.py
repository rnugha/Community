import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def load_documents(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load()

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_documents(documents):
    chunks = []
    pattern = r'(?=\n?\d{1,2}\.\s)'

    for doc in documents:
        parts = re.split(pattern, doc.page_content)
        for part in parts:
            part = clean_text(part)
            if part and len(part) > 1:
                chunks.append(Document(page_content=part, metadata=doc.metadata))
    return chunks

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

def get_vectorstore(chroma_path):
    return Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())

def init_env():
    load_dotenv()

def ingest_pdfs_to_chroma(data_path, chroma_path):
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    db = get_vectorstore(chroma_path)
    db.add_documents(chunks, ids=ids)
    db.persist()