import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_documents(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load()

def extract_chunks(documents):
    chunks = []
    for doc in documents:
        text = doc.page_content
        text = text.replace('\r', ' ')

        pattern = r'(chunk\s*\n*\s*\d+\..*?)(?=chunk\s*\n*\s*\d+\.|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            clean_part = match.strip().replace('\n', ' ')
            chunks.append(Document(page_content=f"passage: {clean_part}", metadata=doc.metadata))
    return chunks

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

def get_vectorstore(chroma_path):
    return Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())

def init_env():
    load_dotenv()

def ingest_pdfs_to_chroma(data_path, chroma_path):
    documents = load_documents(data_path)
    chunks = extract_chunks(documents)
    if not chunks:
        raise ValueError("No chunks extracted! Please check splitting regex.")
    print(f"Extracted {len(chunks)} chunks")
    ids = [f"chunk-{i+1}" for i in range(len(chunks))]
    db = get_vectorstore(chroma_path)
    db.add_documents(chunks, ids=ids)
    db.persist()
