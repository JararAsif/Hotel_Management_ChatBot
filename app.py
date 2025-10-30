import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "docs"

st.set_page_config(page_title="Hotel Assistant Bot — LangChain + Chroma", layout="wide")

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in your .env file.")
    st.stop()

st.title("Hotel Assistant Chatbot — LangChain + Chroma + OpenAI")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    temperature=0.3,
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful and friendly **hotel assistant chatbot**.
You can answer general queries from hotel guests — such as room availability, restaurant timings, nearby attractions, and hotel services.
Always answer politely, clearly, and with accurate information based on the context provided.

<context>
{context}
</context>

Guest question: {question}
""")

retrieval_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

with st.form("query_form"):
    query = st.text_input("Ask your hotel-related question:")
    submitted = st.form_submit_button("Ask")

if submitted and query.strip():
    with st.spinner("Fetching answer..."):
        response = retrieval_chain.invoke(query)
        st.subheader("Answer")
        st.write(response.content)
else:
    st.info("Type your question and press 'Ask'. Ensure you've run `ingest.py` to load hotel documents into Chroma DB.")
