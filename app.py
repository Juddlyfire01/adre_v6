import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
openai.api_base = st.secrets.openai_api_base
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the Approved Document B",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # Standard LlamaIndex approach - simple and efficient
    from llama_index.readers.file import PyMuPDFReader
    import tiktoken
    
    st.write("🚀 Starting document processing...")
    
    # Use PyMuPDF for better PDF text extraction
    reader = PyMuPDFReader()
    docs = reader.load_data(file_path="./data/Approved_Document_B_volume_1_Dwellings_2019_edition_incorporating_2020_2022_and_2025_amendments_collated_with_2026_and_2029_amendments.pdf")
    
    st.write(f"📄 Loaded {len(docs)} documents")
    
    # Show document info
    for i, doc in enumerate(docs):
        doc_size = len(doc.text)
        st.write(f"Document {i}: {doc_size:,} characters")
        if doc_size > 10000:
            st.warning(f"⚠️ Document {i} is very large: {doc_size:,} characters")
    
    # Configure text splitter with much smaller chunks to avoid token limits
    from llama_index.core.node_parser import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    
    st.write("🔪 Splitting documents into chunks...")
    
    # Process documents with logging
    nodes = text_splitter.get_nodes_from_documents(docs)
    st.write(f"📊 Created {len(nodes)} chunks")
    
    # Log chunk sizes and count actual tokens
    chunk_sizes = [len(node.text) for node in nodes]
    st.write(f"📏 Chunk sizes - Min: {min(chunk_sizes):,}, Max: {max(chunk_sizes):,}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
    
    # Count actual tokens using tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    actual_tokens = [len(encoding.encode(node.text)) for node in nodes]
    st.write(f"🎯 Actual tokens - Min: {min(actual_tokens)}, Max: {max(actual_tokens)}, Avg: {sum(actual_tokens)/len(actual_tokens):.0f}")
    
    # Check for oversized chunks
    oversized_chunks = [i for i, tokens in enumerate(actual_tokens) if tokens > 7000]  # Conservative limit
    if oversized_chunks:
        st.error(f"❌ Found {len(oversized_chunks)} chunks that are TOO LARGE: {oversized_chunks[:5]}")
        for i in oversized_chunks[:3]:
            st.write(f"Chunk {i}: {chunk_sizes[i]} chars ({actual_tokens[i]} tokens) - Preview: {nodes[i].text[:200]}...")
    
    # Show first few chunks for debugging
    st.write("🔍 First 3 chunks preview:")
    for i in range(min(3, len(nodes))):
        st.write(f"Chunk {i}: {len(nodes[i].text)} chars ({actual_tokens[i]} tokens) - {nodes[i].text[:100]}...")
    
    # Configure embedding model with smaller batch size
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=st.secrets.openai_key,
        api_base=st.secrets.openai_api_base,
        embed_batch_size=10  # Smaller batch size to avoid token limits
    )
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on 
        building regulations and fire safety. 
        Answer questions based on the provided 
        documentation. Keep your answers technical 
        and based on facts – do not hallucinate regulations.""",
    )
    
    st.write("🔗 Creating vector index with progress tracking...")
    
    # Create index using standard approach with proper text splitting and progress
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[text_splitter],
        show_progress=True
    )
    
    st.write("✅ Index created successfully!")
    st.write(f"📈 Final stats: {len(nodes)} chunks processed, {sum(actual_tokens):,} total tokens")
    
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
