import os
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "docs"

if not OPENAI_API_KEY:
    raise ValueError("Set your OPENAI_API_KEY in .env or environment variables first!")

def load_documents(data_dir="data"):
    """Load all .txt and .pdf documents from the given directory."""
    docs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif file.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
    return docs

def main():
    print("Starting ingestion...")

    docs = load_documents("data")
    if not docs:
        print("No documents found in ./data — add .pdf or .txt files and re-run.")
        return

    print(f"Loaded {len(docs)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings and saving to Chroma...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    vectordb.persist()
    print(f"Successfully saved {len(chunks)} chunks to Chroma at {CHROMA_DB_DIR}")

if __name__ == "__main__":
    main()
