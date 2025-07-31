import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Configure API
import openai
openai.api_key = st.secrets.openai_key
openai.api_base = st.secrets.openai_api_base

# Standard approach
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key=st.secrets.openai_key,
    api_base=st.secrets.openai_api_base
)

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)

# Load documents
reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
docs = reader.load_data()

st.write(f"Loaded {len(docs)} documents")

# Create index
index = VectorStoreIndex.from_documents(docs)
st.write("Index created successfully!")

# Test query
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
st.write(response) 