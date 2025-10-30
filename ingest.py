import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "docs"

if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in environment or .env file")
