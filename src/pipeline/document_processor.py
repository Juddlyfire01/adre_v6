import streamlit as st
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import tiktoken


class DocumentProcessor:
    """Handles document loading, processing, and indexing for the RAG application."""
    
    def __init__(self, openai_api_key: str, openai_api_base: str):
        """Initialize the document processor with API credentials."""
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self._setup_models()
    
    def _setup_models(self):
        """Configure embedding and LLM models."""
        # Configure embedding model
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=self.openai_api_key,
            api_base=self.openai_api_base,
            embed_batch_size=10
        )
        
        # Configure LLM
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="""You are an expert on 
            building regulations and fire safety. 
            Answer questions based on the provided 
            documentation. Keep your answers technical 
            and based on facts – do not hallucinate regulations.""",
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
    
    def load_documents(self, file_path: str):
        """Load documents from the specified file path."""
        st.write("🚀 Starting document processing...")
        
        # Use PyMuPDF for better PDF text extraction
        reader = PyMuPDFReader()
        docs = reader.load_data(file_path=file_path)
        
        st.write(f"📄 Loaded {len(docs)} documents")
        
        # Show document info
        for i, doc in enumerate(docs):
            doc_size = len(doc.text)
            st.write(f"Document {i}: {doc_size:,} characters")
            if doc_size > 10000:
                st.warning(f"⚠️ Document {i} is very large: {doc_size:,} characters")
        
        return docs
    
    def create_text_splitter(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Create a text splitter with specified parameters."""
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def process_documents(self, docs, text_splitter):
        """Process documents into chunks and validate them."""
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
        
        return nodes, actual_tokens
    
    def create_index(self, docs, text_splitter):
        """Create vector index from documents."""
        st.write("🔗 Creating vector index with progress tracking...")
        
        # Create index using standard approach with proper text splitting and progress
        index = VectorStoreIndex.from_documents(
            docs,
            transformations=[text_splitter],
            show_progress=True
        )
        
        st.write("✅ Index created successfully!")
        
        return index
    
    def process_and_index(self, file_path: str):
        """Complete document processing pipeline."""
        # Load documents
        docs = self.load_documents(file_path)
        
        # Create text splitter
        text_splitter = self.create_text_splitter()
        
        # Process documents
        nodes, actual_tokens = self.process_documents(docs, text_splitter)
        
        # Create index
        index = self.create_index(docs, text_splitter)
        
        st.write(f"📈 Final stats: {len(nodes)} chunks processed, {sum(actual_tokens):,} total tokens")
        
        return index


@st.cache_resource(show_spinner=False)
def load_data():
    """Cached function to load and process documents."""
    processor = DocumentProcessor(
        openai_api_key=st.secrets.openai_key,
        openai_api_base=st.secrets.openai_api_base
    )
    
    return processor.process_and_index(
        "./data/Approved_Document_B_volume_1_Dwellings_2019_edition_incorporating_2020_2022_and_2025_amendments_collated_with_2026_and_2029_amendments.pdf"
    ) 