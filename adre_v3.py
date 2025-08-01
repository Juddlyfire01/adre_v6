import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter, SemanticSplitterNodeParser, SentenceWindowNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import Document
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import os
import logging
import json
import fitz  # PyMuPDF
import re
import tiktoken  # Add tiktoken for accurate token counting
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize tiktoken encoder for accurate token counting
try:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
except:
    encoding = tiktoken.get_encoding("cl100k_base")

def get_token_count(text: str) -> int:
    """Get accurate token count using tiktoken"""
    return len(encoding.encode(text))

# ============== SOURCE INJECTION FOR ACCURATE CITATIONS ==============

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode

class SourceInjectionPostprocessor(BaseNodePostprocessor):
    """Modifies chunks to include source information directly in the text."""
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        """Prepend source information to each chunk for accurate citation tracking."""
        
        modified_nodes = []
        
        for i, node in enumerate(nodes, 1):
            # Extract metadata - use metadata directly, don't re-parse text
            meta = node.node.metadata
            
            # Create detailed source header using ONLY metadata (not text parsing)
            source_parts = []
            if meta.get('regulation_ref'):
                source_parts.append(f"Document {meta['regulation_ref']}")
            elif meta.get('document_title'):
                source_parts.append(f"Document {meta['document_title']}")
                
            if meta.get('section', 'Unknown') != 'Unknown':
                source_parts.append(f"Section {meta['section']}")
                
            # Use metadata paragraph number (should be correct with improved extraction)
            if meta.get('paragraph'):
                source_parts.append(f"paragraph {meta['paragraph']}")
                
            if meta.get('internal_page'):
                source_parts.append(f"page {meta['internal_page']}")
            
            source_header = f"Source {i} ({', '.join(source_parts)}):\n"
            
            # Get the current text content
            current_text = node.node.get_content()
            
            # Create new text with source header
            new_text = source_header + current_text
            
            # Create a new TextNode with the modified content
            new_node = TextNode(
                text=new_text,
                metadata=meta,
                id_=node.node.id_
            )
            
            # Create new NodeWithScore with the new node
            modified_node = NodeWithScore(node=new_node, score=node.score)
            modified_nodes.append(modified_node)
            
        return modified_nodes

# ============== NEW SIMPLIFIED PARAGRAPH-BASED CHUNKING ==============

def extract_paragraphs_from_page(page_text: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract paragraphs from a page using the natural structure of Approved Documents.
    Each paragraph becomes one chunk - no splitting, no merging.
    """
    paragraphs = []
    
    # Clean up common PDF artifacts
    page_text = page_text.replace('\n\n', '\n')  # Remove double newlines
    
    # Main paragraph pattern: matches 1.1, 1.28, 2.3A, etc.
    # More flexible to catch variations
    main_para_pattern = r'^(\d+\.\d+[A-Z]?)\s+(.+?)(?=^\d+\.\d+[A-Z]?\s|\Z)'
    
    # Sub-paragraph patterns
    letter_sub_pattern = r'^([a-z])\.\s+(.+?)(?=^[a-z]\.\s|\Z)'  # a. b. c.
    roman_sub_pattern = r'^([ivxIVX]+)\.\s+(.+?)(?=^[ivxIVX]+\.\s|\Z)'  # i. ii. iii.
    
    # Diagram/Figure/Table patterns
    figure_pattern = r'((?:Diagram|Figure|Table)\s*\d+(?:\.\d+)?[a-z]?)'
    
    # Process text to find all paragraphs
    current_pos = 0
    lines = page_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line starts a main paragraph
        # FIXED: More restrictive pattern to avoid matching sub-paragraph references
        # Only match if the paragraph number is followed by actual content (not just "to X.XX)")
        main_match = re.match(r'^(\d+\.\d+[A-Z]?)\s+(.+)', line)
        if main_match:
            para_num = main_match.group(1)
            para_start = main_match.group(2)
            
            # FIXED: Skip if this looks like a sub-paragraph reference
            # (e.g., "4.17 to 4.20);" is not a real paragraph start)
            if re.match(r'^to\s+\d+\.\d+\)', para_start.strip()) or para_start.strip().endswith(');'):
                i += 1
                continue
            
            # Collect the rest of the paragraph
            para_lines = [para_start]
            j = i + 1
            
            # Continue collecting lines until we hit another paragraph or section marker
            while j < len(lines):
                next_line = lines[j].strip()
                
                # Stop if we hit another paragraph, section, or major structural element
                if (re.match(r'^\d+\.\d+[A-Z]?\s', next_line) or 
                    re.match(r'^Section\s+\d+', next_line) or
                    re.match(r'^[A-Z][a-z]+\s+\d+\s*–', next_line) or  # "Section 1 –" format
                    next_line.startswith('Appendix') or
                    not next_line):  # Empty line might indicate end
                    break
                
                # Check for sub-paragraphs within
                if re.match(r'^[a-z]\.\s', next_line) or re.match(r'^[ivxIVX]+\.\s', next_line):
                    # Include sub-paragraphs as part of the main paragraph
                    para_lines.append(next_line)
                else:
                    para_lines.append(next_line)
                
                j += 1
            
            # Join the paragraph text
            full_para_text = '\n'.join(para_lines)
            
            # Extract any sub-paragraphs for metadata
            sub_items = []
            for sub_match in re.finditer(r'^([a-z])\.\s', full_para_text, re.MULTILINE):
                sub_items.append(sub_match.group(1))
            
            # Check for diagrams/figures
            diagram = None
            fig_match = re.search(figure_pattern, full_para_text, re.IGNORECASE)
            if fig_match:
                diagram = fig_match.group(1)
            
            paragraphs.append({
                'number': para_num,
                'text': full_para_text,
                'sub_items': sub_items,
                'diagram': diagram,
                'page': page_num,
                'start_line': i,
                'end_line': j - 1
            })
            
            i = j
        else:
            # FIXED: Handle paragraph numbers on their own line (like "4.1\t")
            if re.match(r'^\d+\.\d+[A-Z]?\s*$', line):
                # Check if the next line has content
                if i + 1 < len(lines) and lines[i + 1].strip():
                    para_num = line.strip()
                    para_start = lines[i + 1].strip()
                    
                    # Skip if this looks like a sub-paragraph reference
                    if re.match(r'^to\s+\d+\.\d+\)', para_start.strip()) or para_start.strip().endswith(');'):
                        i += 1
                        continue
                    
                    # Collect the rest of the paragraph
                    para_lines = [para_start]
                    j = i + 2
                    
                    # Continue collecting lines until we hit another paragraph or section marker
                    while j < len(lines):
                        next_line = lines[j].strip()
                        
                        # Stop conditions - but exclude false paragraph patterns
                        if re.match(r'^\d+\.\d+[A-Z]?\s', next_line):
                            # Check if this is a real paragraph or false one
                            false_para_match = re.match(r'^(\d+\.\d+[A-Z]?)\s+(.+)', next_line)
                            if false_para_match:
                                false_start = false_para_match.group(2)
                                if not (re.match(r'^to\s+\d+\.\d+\)', false_start.strip()) or false_start.strip().endswith(');')):
                                    # This is a real paragraph, stop here
                                    break
                        elif re.match(r'^\d+\.\d+[A-Z]?\s*$', next_line):
                            # Another paragraph number on its own line
                            break
                        elif (re.match(r'^Section\s+\d+', next_line) or
                              re.match(r'^[A-Z][a-z]+\s+\d+\s*–', next_line) or
                              next_line.startswith('Appendix') or
                              not next_line):
                            break
                        
                        # Check for sub-paragraphs within
                        if re.match(r'^[a-z]\.\s', next_line) or re.match(r'^[ivxIVX]+\.\s', next_line):
                            # Include sub-paragraphs as part of the main paragraph
                            para_lines.append(next_line)
                        else:
                            para_lines.append(next_line)
                        
                        j += 1
                    
                    # Join the paragraph text
                    full_para_text = '\n'.join(para_lines)
                    
                    # Extract any sub-paragraphs for metadata
                    sub_items = []
                    for sub_match in re.finditer(r'^([a-z])\.\s', full_para_text, re.MULTILINE):
                        sub_items.append(sub_match.group(1))
                    
                    # Check for diagrams/figures
                    diagram = None
                    fig_match = re.search(figure_pattern, full_para_text, re.IGNORECASE)
                    if fig_match:
                        diagram = fig_match.group(1)
                    
                    paragraphs.append({
                        'number': para_num,
                        'text': full_para_text,
                        'sub_items': sub_items,
                        'diagram': diagram,
                        'page': page_num,
                        'start_line': i,
                        'end_line': j - 1
                    })
                    
                    i = j
                else:
                    i += 1
            else:
                i += 1
    
    return paragraphs

def create_paragraph_based_chunks(pdf_path: str, toc: List) -> List[Document]:
    """
    Create chunks using the natural paragraph structure of Approved Documents.
    One paragraph = one chunk. Simple, clean, accurate.
    """
    doc = fitz.open(pdf_path)
    toc_tree = build_toc_tree(toc)
    all_chunks = []
    
    logger.info("Starting simplified paragraph-based chunking...")
    
    for pdf_page_num in range(len(doc)):
        page = doc[pdf_page_num]
        page_text = page.get_text()
        
        if not page_text.strip():
            continue
        
        # Get TOC context for this page
        pdf_page = pdf_page_num + 1
        toc_node = find_deepest_node(toc_tree, pdf_page)
        
        # Extract metadata from TOC
        metadata = extract_enhanced_metadata(toc_node, pdf_page, pdf_page_num, page_text)
        
        # Extract paragraphs from this page
        paragraphs = extract_paragraphs_from_page(page_text, pdf_page)
        
        # Create one chunk per paragraph
        for para in paragraphs:
            # FIXED: Extract metadata specifically for this paragraph's content
            # instead of using page-level metadata which only looks at the first 200 chars
            para_metadata = extract_enhanced_metadata(toc_node, pdf_page, pdf_page_num, para['text'])
            
            # Use the paragraph number from the extraction (most reliable)
            # but fall back to para_metadata if needed
            correct_paragraph = para['number'] or para_metadata.paragraph
            
            # Create clean metadata for this paragraph
            chunk_metadata = {
                "document_title": metadata.document_title,  # Keep page-level document info
                "section": metadata.section,  # Keep page-level section info  
                "subsection": metadata.subsection or "",
                "paragraph": correct_paragraph,  # Use paragraph-specific extraction
                "internal_page": metadata.internal_page,
                "pdf_page": metadata.pdf_page,
                "diagram": para.get('diagram', "") or para_metadata.diagram or "",
                "regulation_ref": metadata.regulation_ref or para_metadata.regulation_ref or "",
                "requirement_ref": para_metadata.requirement_ref or "",
                "sub_items": para.get('sub_items', []),
                "text_type": "paragraph",  # Simple type
                "chunk_id": f"para_{correct_paragraph}_p{pdf_page}"
            }
            
            # Check token count
            token_count = get_token_count(para['text'])
            
            # Most paragraphs should be reasonable size, but log if not
            if token_count > 1024:
                logger.warning(f"Large paragraph {para['number']} on page {pdf_page}: {token_count} tokens")
            
            # Create the chunk
            chunk = Document(
                text=para['text'],
                metadata=chunk_metadata
            )
            
            all_chunks.append(chunk)
        
        if pdf_page_num % 50 == 0:
            logger.info(f"Processed {pdf_page_num + 1}/{len(doc)} pages, extracted {len(all_chunks)} paragraphs")
    
    logger.info(f"Paragraph extraction complete! Created {len(all_chunks)} chunks")
    
    # Log statistics
    token_counts = [get_token_count(chunk.text) for chunk in all_chunks]
    if token_counts:
        logger.info(f"Token count stats - Min: {min(token_counts)}, Max: {max(token_counts)}, "
                   f"Avg: {sum(token_counts)/len(token_counts):.1f}")
    
    return all_chunks

def create_simplified_citation(meta: Dict[str, Any]) -> str:
    """
    Create a simple, clean citation from paragraph-based metadata.
    This replaces the complex citation system with direct mapping.
    """
    parts = []
    
    # Document/Part reference - properly handle single letter documents
    if meta.get('regulation_ref'):
        # Clean up regulation reference - remove "Part" if it's just a letter
        reg_ref = meta['regulation_ref'].strip()
        if len(reg_ref) == 1 and reg_ref.isalpha():
            parts.append(f"Document {reg_ref}")
        else:
            parts.append(f"Part {reg_ref}")
    elif meta.get('document_title', 'Unknown') != 'Unknown':
        # Extract part from document title
        doc_title = meta['document_title']
        # Look for single letter documents (like "C")
        if doc_title.strip() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']:
            parts.append(f"Document {doc_title.strip()}")
        else:
            part_match = re.search(r'Part\s+([A-Z]\d*)', doc_title, re.IGNORECASE)
            if part_match:
                parts.append(f"Part {part_match.group(1)}")
    
    # Section
    if meta.get('section', 'Unknown') != 'Unknown':
        section = meta['section']
        # Clean up section - remove redundant "Section" prefix
        section = re.sub(r'^Section\s+', '', section).strip()
        # Don't add "Section" again if it's already in the text
        if section and not section.lower().startswith('section'):
            parts.append(f"Section {section}")
        elif section:
            parts.append(section)
    
    # Paragraph - the most important part
    if meta.get('paragraph'):
        parts.append(f"paragraph {meta['paragraph']}")
    
    # Page
    if meta.get('internal_page'):
        parts.append(f"page {meta['internal_page']}")
    
    return f"{', '.join(parts)}" if parts else "Source unavailable"

# ============== END NEW SIMPLIFIED CHUNKING ==============

def extract_document_reference_from_query(query: str) -> Optional[str]:
    """
    Extract document reference from user query to enable document filtering.
    Returns the document letter (A, B, C, etc.) if found in the query.
    """
    # Patterns to match document references in queries
    doc_patterns = [
        r'\b(?:Approved\s+)?Document\s+([A-Z])\b',  # "Document A", "Approved Document A"
        r'\bPart\s+([A-Z])\b',  # "Part A", "Part B"
        r'\b([A-Z])\s+(?:document|part|regulation)\b',  # "A document", "B part"
        r'\b(?:document|part)\s+([A-Z])\b',  # "document A", "part B"
    ]
    
    for pattern in doc_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def create_document_filtered_query_engine(index, target_document: str):
    """
    Create a query engine that filters results to a specific document.
    """
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    
    # Create a custom retriever that filters by document
    class DocumentFilteredRetriever(VectorIndexRetriever):
        def __init__(self, index, target_document: str, **kwargs):
            super().__init__(index, **kwargs)
            self.target_document = target_document
        
        def retrieve(self, query_bundle):
            # Get all nodes from the base retriever
            all_nodes = super().retrieve(query_bundle)
            
            # Filter nodes to only include those from the target document
            filtered_nodes = []
            for node in all_nodes:
                meta = getattr(node.node, "metadata", {})
                regulation_ref = meta.get('regulation_ref', '')
                document_title = meta.get('document_title', '')
                
                # Check if this node belongs to the target document
                if (regulation_ref == self.target_document or 
                    document_title == self.target_document or
                    (len(document_title) == 1 and document_title == self.target_document)):
                    filtered_nodes.append(node)
            
            return filtered_nodes
    
    # Create the filtered retriever
    retriever = DocumentFilteredRetriever(
        index, 
        target_document,
        similarity_top_k=similarity_top_k,
        node_postprocessors=[SourceInjectionPostprocessor()]
    )
    
    # Create query engine with the filtered retriever
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=index.get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            citation_qa_template=CITATION_PROMPT
        )
    )

# Initialize chat history
if "messages_v2" not in st.session_state:
    st.session_state.messages_v2 = [
        {"role": "assistant", "content": "A.D.R.E Ready.", "sources": None, "citation_map": None}
    ]

# Add chat history for chat mode
if "chat_mode_history" not in st.session_state:
    st.session_state.chat_mode_history = [
        {"role": "assistant", "content": "A.D.R.E Ready."}
    ]

# Add generating flags to track when responses are being generated
if "generating_reference" not in st.session_state:
    st.session_state.generating_reference = False
if "generating_chat" not in st.session_state:
    st.session_state.generating_chat = False

# Add chat engine mode tracking
if "current_chat_mode" not in st.session_state:
    st.session_state.current_chat_mode = "condense_plus_context"

# Add citation mode tracking
if "use_full_citations" not in st.session_state:
    st.session_state.use_full_citations = False

# Add mode tracking
if "selected_mode_key" not in st.session_state:
    st.session_state.selected_mode_key = "reference"

# Create query engine - simplified for paragraph-based chunks
@st.cache_resource
def get_query_engine(_index, top_k, chunk_size, use_context):
    # Only 'compact' response mode is supported
    response_mode_enum = ResponseMode.COMPACT
    return CitationQueryEngine.from_args(
        _index,
        similarity_top_k=top_k,
        citation_chunk_size=chunk_size,
        citation_qa_template=CITATION_PROMPT,
        node_postprocessors=[SourceInjectionPostprocessor()],  # Add source injection
        output_cls=None,
        response_mode=response_mode_enum
    )

# Create chat engine with dynamic mode selection
@st.cache_resource
def get_chat_engine(_index, chat_mode="condense_plus_context"):
    # Import ChatMemoryBuffer for conversation memory management
    from llama_index.core.memory import ChatMemoryBuffer
    
    # Configure memory with appropriate token limit for regulatory content
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
    # Define prompts for different modes
    prompts = {
        "condense_plus_context": (
            "You are an expert on Building Regulations for England, able to have natural conversations "
            "while providing authoritative answers about building regulations and approved documents. "
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history and the context above to provide accurate, "
            "helpful responses. When using information from the context, add numbered citations [1], [2], etc. "
            "immediately after factual statements. Each number corresponds to the source order in the context. "
            "Be conversational but maintain technical accuracy."
        ),
        "context": (
            "You are an expert on Building Regulations for England. "
            "Answer questions conversationally, but whenever you use information from the provided context, "
            "add a numbered citation [1], [2], etc. immediately after the relevant statement. "
            "Each number should correspond to the order of the source in the context. "
            "If you do not use any information from the context, do not add a citation. "
            "Do not fabricate citations."
        ),

        "best": (
            "You are an expert on Building Regulations for England, able to have natural conversations "
            "while providing authoritative answers about building regulations and approved documents. "
            "When using information from retrieved context, add numbered citations [1], [2], etc. "
            "immediately after factual statements. Be conversational but maintain technical accuracy."
        ),
        "condense_question": (
            "You are an expert on Building Regulations for England. "
            "Answer questions based on the provided context, adding numbered citations [1], [2], etc. "
            "when using information from the sources. Be precise and technical."
        )
    }
    
    # Get the appropriate prompt
    prompt = prompts.get(chat_mode, prompts["condense_plus_context"])
    
    # Configure chat engine based on mode
    if chat_mode in ["condense_plus_context", "condense_question"]:
        return _index.as_chat_engine(
            chat_mode=chat_mode,
            memory=memory,
            llm=Settings.llm,
            context_prompt=prompt,
            verbose=True
        )
    elif chat_mode == "context":
        return _index.as_chat_engine(
            chat_mode=chat_mode,
            llm=Settings.llm,
            system_prompt=prompt,
            verbose=True
        )

    elif chat_mode == "best":
        return _index.as_chat_engine(
            chat_mode=chat_mode,
            memory=memory,
            llm=Settings.llm,
            system_prompt=prompt,
            verbose=True
        )
    else:
        # Fallback to condense_plus_context
        return _index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            llm=Settings.llm,
            context_prompt=prompts["condense_plus_context"],
            verbose=True
        )

# Streamlit configuration
st.set_page_config(
    page_title="A.D.R.E - Approved Document Reference Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("A.D.R.E - Approved Document Reference Engine")
st.markdown(
    '[Source File: The_Merged_Approved_Documents_Oct24.pdf](https://assets.publishing.service.gov.uk/media/6717d29438149ce9d09e3862/The_Merged_Approved_Documents_Oct24.pdf)',
    unsafe_allow_html=True
)

# Enhanced sidebar configuration
with st.sidebar:
    # Mode selector in an expander
    with st.expander("Mode", expanded=False):
        mode = st.radio(
            "Select mode",
            options=["Reference", "Consultation"],
            index=0 if st.session_state.selected_mode_key == "reference" else 1,
            key="mode_radio",
            help=(
                "Reference: Precise, citation-focused answers with detailed source references. "
                "Best for technical compliance questions and regulatory research.\n\n"
                "Consultation: Conversational AI that remembers your previous questions. "
                "Best for ongoing discussions and follow-up queries."
            ),
            format_func=lambda x: "Reference" if x == "Reference" else "Consultation"
        )
        st.session_state.selected_mode_key = "reference" if mode == "Reference" else "chat"
        
        # Chat Engine Mode Selector (only show in consultation mode)
        if st.session_state.selected_mode_key == "chat":
            st.markdown("---")
            st.markdown("**Consultation Engine Type**")
            chat_engine_modes = {
                "Condense + Context (Recommended)": "condense_plus_context",
                "Condense Question": "condense_question",
                "Context Only": "context",
                "Best Mode (Auto-select)": "best"
            }
            
            # Find current mode index
            current_mode_display = None
            for display_name, mode_key in chat_engine_modes.items():
                if mode_key == st.session_state.current_chat_mode:
                    current_mode_display = display_name
                    break
            
            current_index = list(chat_engine_modes.keys()).index(current_mode_display) if current_mode_display else 0
            
            selected_chat_mode = st.selectbox(
                "Engine",
                list(chat_engine_modes.keys()),
                index=current_index,
                help=(
                    "• **Condense + Context**: Best for follow-up questions and conversation flow\n"
                    "• **Condense Question**: Condenses questions but simpler than Condense + Context\n"
                    "• **Context Only**: Uses retrieved context without conversation memory\n" 
                    "• **Best Mode**: Automatically selects the best approach"
                )
            )
            selected_chat_mode_key = chat_engine_modes[selected_chat_mode]
            
            # Check if mode changed and clear cache if so
            if selected_chat_mode_key != st.session_state.current_chat_mode:
                st.session_state.current_chat_mode = selected_chat_mode_key
                # Clear the chat engine cache to force recreation with new mode
                try:
                    get_chat_engine.clear()
                except NameError:
                    # Function not defined yet, which is fine - it will be created with the new mode
                    pass
                st.info(f"Switched to {selected_chat_mode} mode. Previous conversation context may be lost.")
                st.rerun()

    # Settings in an expander
    with st.expander("Settings", expanded=False):
        section_style = "font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.25rem;"
        field_style = "font-size: 1rem; font-weight: 400; margin-bottom: 0.25rem;"
        
        st.markdown(f'<span style="{field_style}">Reference Depth</span>', unsafe_allow_html=True)
        similarity_top_k = st.slider(
            "Reference Depth",  # Non-empty label for accessibility
            min_value=3,
            max_value=15,
            value=5,
            help=(
                "This controls the 'top-k' parameter: how many of the most relevant references (paragraphs) are retrieved from the Approved Documents for each query. "
                "\n\n"
                "A lower value (e.g., 3) means only the most relevant references are shown—this can make answers more focused but risks missing supporting context. "
                "A higher value (e.g., 15) increases the reference depth, which may provide broader context but can introduce less relevant or redundant information. "
                "\n\n"
                "For ADRE, a moderate value (like 5) is usually best: it balances precision and context, ensuring answers are well-supported without overwhelming you with too much information."
            )
        )
        
        st.markdown("---")
        st.markdown(f'<span style="{field_style}">Citation Format</span>', unsafe_allow_html=True)
        use_full_citations = st.toggle(
            "Full Inline Citations",
            value=st.session_state.use_full_citations,
            help=(
                "**Full Inline Citations (ON):** Shows complete source information inline like '(Document A, Section 2.1, paragraph 3, page 45)'\n\n"
                "**Numbered Citations (OFF):** Uses standard academic format with [1], [2], [3] and a reference list below"
            )
        )
        
        # Update session state if changed
        if use_full_citations != st.session_state.use_full_citations:
            st.session_state.use_full_citations = use_full_citations
            # Clear chat history when citation format changes to avoid mixed formats
            st.session_state.messages_v2 = [
                {"role": "assistant", "content": "A.D.R.E Ready.", "sources": None, "citation_map": None}
            ]
            st.session_state.chat_mode_history = [
                {"role": "assistant", "content": "A.D.R.E Ready."}
            ]
        
        citation_chunk_size = 1024  # Fixed value since paragraphs are natural chunks
        use_window_context = False  # Not needed with paragraph chunks

    # About section in an expander
    with st.expander("About", expanded=False):
        st.markdown(f'<span style="{section_style}">A.D.R.E – Approved Document Reference Engine</span>', unsafe_allow_html=True)
        st.image("app.png", use_container_width=True)
        st.markdown("""
A.D.R.E advances how construction professionals access and apply building regulations. Instantly search official documents, get precise, paragraph-level citations, and ensure compliance with up-to-date standards—all in one secure, easy-to-use tool.

- **For engineers & compliance teams:** Save time, reduce risk, and support technical decisions with authoritative references.
- **Transforms manual referencing:** No more page-flipping or citation errors—A.D.R.E delivers the exact requirement you need, when you need it.
- **Powered by advanced AI workflows:** Uses advanced language models to break down, index, and search the latest Approved Documents.
        """)

    # Documentation section in an expander
    with st.expander("Documentation", expanded=False):
        try:
            with open("TECHNICAL_ARCHITECTURE.md", "r", encoding="utf-8") as f:
                tech_doc = f.read()
            # Apply the same styling as the About section
            styled_doc = f'<div style="{section_style}">{tech_doc}</div>'
            st.markdown(styled_doc, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Technical architecture documentation not found.")
        except Exception as e:
            st.error(f"Error loading documentation: {e}")

    # Benchmark section in an expander
    with st.expander("Benchmark", expanded=False):
        st.markdown(f'<span style="{section_style}">A.D.R.E Dynamic Test Results</span>', unsafe_allow_html=True)
        
        # Key Features
        st.markdown("**Key Features:**")
        st.markdown("""
        • **Dynamic Question Generation**: LLM-based conversion of regulatory paragraphs to test questions
        • **Metadata Comparison**: Direct comparison between original and retrieved source metadata
        • **Source Validation**: Tests ADRE's ability to find source of randomly selected content
        • **Citation Accuracy**: Validates citation references and formatting
        • **Performance Metrics**: Response time and token efficiency tracking
        """)
        
        # Results Summary
        st.markdown("**Test Results Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Metadata Match Score", "0.988", help="How well retrieved sources match original metadata")
            st.metric("Source Relevance", "0.981", help="Whether retrieved sources contain the original paragraph")
        with col2:
            st.metric("Citation Accuracy", "0.999", help="Accuracy of citations in responses")
            st.metric("Avg Response Time", "3.18s", help="Average time to generate responses")
        
        # Implications
        st.markdown("**Regulatory Compliance Implications:**")
        st.markdown("""
        • **High Metadata Match (98.8%)**: ADRE accurately identifies the correct regulatory source, crucial for compliance verification
        • **Excellent Source Relevance (98.1%)**: Retrieved content directly relates to original requirements, ensuring regulatory accuracy
        • **Near-Perfect Citation Accuracy (99.9%)**: Citations are properly formatted and reference valid sources, essential for audit trails
        • **Fast Response Time (3.18s)**: Quick access to regulatory information supports real-time compliance decisions
        """)
        
        # Example Results
        st.markdown("**Example Test Cases:**")
        
        # Example 1
        with st.expander("Example 1: Fire Safety Classification", expanded=False):
            st.markdown("**Original Paragraph:**")
            st.markdown("The surface linings of walls and ceilings should meet the classifications in Table 4.1. Small rooms of maximum internal floor area of 4m²: D-s3, d2. Garages (as part of a dwellinghouse) of maximum internal floor area of 40m²: C-s3, d2.")
            
            st.markdown("**Generated Question:**")
            st.markdown("What is the required classification for the surface linings of walls and ceilings in a garage that is part of a dwellinghouse and has an internal floor area of 30m²?")
            
            st.markdown("**ADRE Response:**")
            st.markdown("The required classification for the surface linings of walls and ceilings in a garage that is part of a dwellinghouse with an internal floor area of 30m² should meet the classification of C-s3, d2 [1]. Since the garage exceeds the 4m² limit for small rooms but is still considered part of a dwellinghouse, and given that garages as part of a dwellinghouse (up to 40m²) fall under 'Other rooms (including garages)', the correct classification is C-s3, d2.")
            
            st.markdown("**Source References:**")
            st.markdown("[1] Document B1, Section 4: Wall and ceiling linings, paragraph 4.1 (PDF Page 127) [2] Document B2, Section 6: Wall and ceiling linings, paragraph 6.1 (PDF Page 318)")
        
        # Example 2
        with st.expander("Example 2: Protected Electrical Circuits", expanded=False):
            st.markdown("**Original Paragraph:**")
            st.markdown("A protected circuit to operate equipment during a fire should achieve all of the following. a. Cables should achieve PH 30 classification when tested in accordance with BS EN 50200 (incorporating Annex E) or an equivalent standard. b. It should only pass through parts of the building in which the fire risk is negligible. c. It should be separate from any circuit provided for another purpose.")
            
            st.markdown("**Generated Question:**")
            st.markdown("What are the requirements for a protected circuit intended to operate equipment during a fire, according to building regulations, in terms of cable classification, routing, and separation from other circuits?")
            
            st.markdown("**ADRE Response:**")
            st.markdown("A protected circuit to operate equipment during a fire should achieve all of the following [1][2]: * Cables should achieve PH 30 classification when tested in accordance with BS EN 50200 (incorporating Annex E) or an equivalent standard [1][2]. * It should only pass through parts of the building in which the fire risk is negligible [1][2]. * It should be separate from any circuit provided for another purpose [1][2]. Note: The same requirements are stated in both Source 1 (Document B2, Section 5: General provisions, paragraph 5.30, page 314) [1] and Source 2 (Document B1, Section 3: Means of escape – flats, paragraph 3.47, page 115) [2].")
            
            st.markdown("**Source References:**")
            st.markdown("[1] Document B2, Section 5: General provisions, paragraph 5.30 (PDF Page 314) [2] Document B1, Section 3: Means of escape – flats, paragraph 3.47 (PDF Page 115)")
        
        # Example 3
        with st.expander("Example 3: Gas-Fired Warm Air System Installation", expanded=False):
            st.markdown("**Original Paragraph:**")
            st.markdown("If a gas-fired warm air system is installed in an existing dwelling, all of the following should be met. a. The system should be installed in accordance with BS 5864. b. All new or replacement ductwork should be insulated in accordance with BS 5422. c. Where controls are external to the heater, the system should be provided with a time switch/programmer and room thermostat, or programmable room thermostat. d. Where controls are integrated in the heater, the system should be provided with a time switch/programmer and room temperature sensor linked to heater firing and fan speed control. e. Independent temperature control of the hot water circuit should be implemented with a cylinder thermostat and a timing device. When there is no demand for hot water both the pump and circulator should switch off.")
            
            st.markdown("**Generated Question:**")
            st.markdown("What specific requirements must be met for the installation of a gas-fired warm air system in an existing dwelling, particularly with regards to controls and temperature regulation, in accordance with relevant British Standards?")
            
            st.markdown("**ADRE Response:**")
            st.markdown("For the installation of a gas-fired warm air system in an existing dwelling, the following specific requirements must be met, particularly with regards to controls and temperature regulation, in accordance with relevant British Standards: * The system should be installed in accordance with BS 5864 [1]. * All new or replacement ductwork should be insulated in accordance with BS 5422 [1]. * Where controls are external to the heater, the system should be provided with a time switch/programmer and room thermostat, or programmable room thermostat [1]. * Where controls are integrated in the heater, the system should be provided with a time switch/programmer and room temperature sensor linked to heater firing and fan speed control [1]. * Independent temperature control of the hot water circuit should be implemented with a cylinder thermostat and a timing device [1]. When there is no demand for hot water, both the pump and circulator should switch off [1].")
            
            st.markdown("**Source References:**")
            st.markdown("[1] Document L1, Section 6, paragraph 6.4 (PDF Page 1023) [2] Document L1, Section 6, paragraph 6.7 (PDF Page 1023) [3] Document L1, Section 6, paragraph 6.5 (PDF Page 1023) [4] Document L1, Section 6, paragraph 6.2 (PDF Page 1022) [5] Document L1, Section 6, paragraph 6.1 (PDF Page 1021)")
        
        # CSV Download
        st.markdown("---")
        st.markdown("**Download Test Results:**")
        try:
            with open("sample_test_results.json", "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            # Convert to CSV format
            csv_data = []
            for result in test_data:
                csv_data.append({
                    'Question ID': result.get('question_id', ''),
                    'Generated Question': result.get('generated_question', ''),
                    'ADRE Answer': result.get('adre_response', 'N/A'),
                    'Metadata Match': f"{result.get('metadata_match_score', 0):.3f}",
                    'Source Relevance': f"{result.get('source_relevance_score', 0):.3f}",
                    'Citation Accuracy': f"{result.get('citation_accuracy_score', 0):.3f}",
                    'Response Time (s)': f"{result.get('response_time', 0):.2f}",
                    'Token Count': result.get('token_count', 0)
                })
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download Test Results CSV",
                data=csv,
                file_name="adre_dynamic_test_results.csv",
                mime="text/csv"
            )
            
            # JSON Download
            json_data = json.dumps(test_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download Test Results JSON",
                data=json_data,
                file_name="adre_dynamic_test_results.json",
                mime="application/json"
            )
        except FileNotFoundError:
            st.error("Test results file not found.")
        except Exception as e:
            st.error(f"Error loading test results: {e}")

# API Configuration
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
os.environ["OPENAI_API_BASE"] = st.secrets.openai_api_base

# Enhanced citation-focused prompt template
CITATION_PROMPT = PromptTemplate(
    """You are an expert on Building Regulations for England. Answer the question based STRICTLY on the provided reference material.

Reference material from relevant documents:
-----------------
{context_str}
-----------------

Question: {query_str}

CITATION REQUIREMENTS:
1. ONLY cite factual statements, requirements, specifications, and technical content from the reference material
2. Use numbered citations [1], [2], etc. immediately after factual statements from the sources
3. If a fact comes from Source 1, cite it as [1]. From Source 2, cite as [2], etc.
4. Multiple sources for one fact: use [1][2] or [1][3]
5. Do NOT cite conversational statements, acknowledgments, or meta-commentary

WHAT TO CITE:
✓ Technical requirements: "The minimum ceiling height shall be 2.3m [1]"
✓ Specifications: "Insulation must have a U-value of 0.16 W/m²K [2]"  
✓ Procedures: "A risk assessment should be undertaken [3]"
✓ Definitions: "A dwelling is defined as a self-contained residential unit [1]"

WHAT NOT TO CITE:
✗ Conversational responses: "I can help you with that question"
✗ Meta-statements: "Based on the documents provided"
✗ Acknowledgments: "I don't have information about that topic"

RESPONSE REQUIREMENTS:
- Answer ONLY using the provided reference material
- Use exact terminology from documents - do not paraphrase technical content
- Be technical, precise, and comprehensive
- Break complex answers into bullet points or lists
- Quote specific values, measurements, and requirements exactly
- If you cannot answer from the reference material, simply state that clearly without false citations

CITATION EXAMPLES:
✓ CORRECT: "Gas control measures for dwellings consist of a gas resistant barrier [1]. The barrier must be across the whole footprint [1]."
✓ CORRECT: "I cannot find information about that specific requirement in the provided documents."
✗ WRONG: "I cannot find information about that specific requirement in the provided documents [1][2][3]."

Answer:"""
)

@dataclass
class ToCNode:
    level: int
    title: str
    page_num: int
    parent: Optional['ToCNode'] = None
    children: List['ToCNode'] = field(default_factory=list)

@dataclass
class DocumentMetadata:
    document_title: str = "Unknown"
    section: str = "Unknown"
    subsection: Optional[str] = None
    paragraph: Optional[str] = None
    internal_page: Optional[int] = None
    pdf_page: Optional[int] = None
    diagram: Optional[str] = None
    regulation_ref: Optional[str] = None
    requirement_ref: Optional[str] = None
    list_context: Optional[str] = None
    text_type: str = "standard"  # semantic, sentence, hybrid

def build_toc_tree(toc: List) -> ToCNode:
    """Build hierarchical table of contents tree"""
    root = ToCNode(0, "ROOT", -1)
    stack = [root]
    
    for level, title, page_num in toc:
        node = ToCNode(level, title, page_num)
        
        # Find the correct parent
        while stack and stack[-1].level >= level:
            stack.pop()
        
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        
        stack.append(node)
    
    return root

def find_deepest_node(node: ToCNode, page: int) -> Optional[ToCNode]:
    """Find the deepest node in TOC tree for a given page"""
    best = None
    for child in node.children:
        if child.page_num <= page:
            best = child
        else:
            break
    
    if best:
        deeper = find_deepest_node(best, page)
        return deeper if deeper else best
    
    return node if node.page_num <= page else None

def extract_enhanced_metadata(toc_node: Optional[ToCNode], page_num: int, pdf_page_num: int, chunk_text: str) -> DocumentMetadata:
    """Extract comprehensive metadata from document chunk with enhanced regulatory detection"""
    
    # Extract document hierarchy from TOC with consistent formatting
    document = section = subsection = "Unknown"
    regulation_ref = None
    
    if toc_node:
        doc_node = section_node = subsection_node = None
        node = toc_node
        
        while node:
            if node.level == 1 and not doc_node:
                doc_node = node
            elif node.level == 2 and not section_node:
                section_node = node
            elif node.level == 3 and not subsection_node:
                subsection_node = node
            node = node.parent
        
        # Standardize document title and regulation reference
        if doc_node:
            document = doc_node.title
            # Extract consistent regulation reference
            # Handle single-letter documents first (most common case)
            if len(document.strip()) == 1 and document.strip().isalpha():
                regulation_ref = document.strip().upper()
            else:
                # Handle complex document titles
                reg_patterns = [
                    r'\bPart\s+([A-Z])\b',  # "Part A", "Part B", etc. - FIXED: very specific
                    r'Approved Document\s+([A-Z]\d*)',  # "Approved Document A"
                    r'^([A-Z]\d+)',        # "R1", "L1" at start (requires digit) - lowest priority
                ]
                for pattern in reg_patterns:
                    match = re.search(pattern, document, re.IGNORECASE)
                    if match:
                        regulation_ref = match.group(1)
                        break
        
        # Check subsection nodes for "Part A" references (higher priority than document title)
        if subsection_node:
            subsection_title = subsection_node.title
            part_match = re.search(r'\bPart\s+([A-Z])\b', subsection_title, re.IGNORECASE)
            if part_match:
                regulation_ref = part_match.group(1)
        
        # Clean up section formatting (remove duplication)
        if section_node:
            section_raw = section_node.title
            # Remove duplicate "Section" words
            section = re.sub(r'\bSection\s+Section\s+', 'Section ', section_raw)
            section = re.sub(r'^Section\s+', '', section)  # Remove leading "Section"
            
        subsection = subsection_node.title if subsection_node else None
    
    # Enhanced pattern matching for regulatory elements
    # Look for paragraph number at the beginning of the chunk or after a newline
    paragraph_patterns = [
        r'(?:^|\n)\s*(\d+\.\d+[A-Z]?)\s+[A-Z]',  # Paragraph at start of line (including 1.3A)
        r'(?:^|\n)\s*(\d+\.\d+\.\d+[A-Z]?)\s+[A-Z]',  # Sub-paragraph at start of line
        r'\bparagraph\s+(\d+\.\d+[A-Z]?)', # Explicit paragraph references
        r'\bClause\s+(\d+\.\d+[A-Z]?)',    # Clause references
    ]
    
    paragraph = None
    # First try to find paragraph at beginning of chunk
    first_200_chars = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
    for pattern in paragraph_patterns[:2]:  # Only check start-of-line patterns
        match = re.search(pattern, first_200_chars, re.MULTILINE)
        if match:
            paragraph = match.group(1)
            break
    
    # If not found at beginning, look for explicit references
    if not paragraph:
        for pattern in paragraph_patterns[2:]:  # Check explicit reference patterns
            match = re.search(pattern, chunk_text, re.IGNORECASE)
            if match:
                paragraph = match.group(1)
                break
    
    # Enhanced diagram/figure/table detection
    diagram_patterns = [
        r'(Diagram\s*\d+\.\d+[a-z]?)',
        r'(Figure\s*\d+\.\d+[a-z]?)',
        r'(Table\s*\d+\.\d+[a-z]?)',
        r'(Appendix\s*[A-Z])',
        r'(Schedule\s*\d+)',
    ]
    
    diagram = None
    for pattern in diagram_patterns:
        match = re.search(pattern, chunk_text, re.IGNORECASE)
        if match:
            diagram = match.group(1)
            break
    
    # Override regulation reference if found in text (more reliable)
    # FIXED: Only match "Part" when it's clearly a document reference, not common usage
    # Look for "Part" followed by a single capital letter (A, B, C, etc.) - this is the document format
    text_regulation_match = re.search(r'\bPart\s+([A-Z])\b', chunk_text)
    if text_regulation_match:
        regulation_ref = text_regulation_match.group(1)
    
    # Also check for "Approved Document A" references in text
    approved_doc_match = re.search(r'Approved Document\s+([A-Z])\b', chunk_text, re.IGNORECASE)
    if approved_doc_match:
        regulation_ref = approved_doc_match.group(1)
    
    requirement_match = re.search(r'Requirement\s+([A-Z]\d+)', chunk_text, re.IGNORECASE)
    requirement_ref = requirement_match.group(1) if requirement_match else None
    
    # Detect list context (assumptions, provisions, requirements)
    list_context = None
    if re.search(r'assumptions?|following\s+assumptions?', chunk_text, re.IGNORECASE):
        list_context = "assumptions"
    elif re.search(r'provisions?|following\s+provisions?', chunk_text, re.IGNORECASE):
        list_context = "provisions"
    elif re.search(r'requirements?|following\s+requirements?', chunk_text, re.IGNORECASE):
        list_context = "requirements"
    
    return DocumentMetadata(
        document_title=document,
        section=section,
        subsection=subsection,
        paragraph=paragraph,
        internal_page=page_num,
        pdf_page=pdf_page_num + 1,
        diagram=diagram,
        regulation_ref=regulation_ref,
        requirement_ref=requirement_ref,
        list_context=list_context
    )

def create_hybrid_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Create hybrid chunks using semantic splitting + sentence windows with strict size constraints:
    - All chunks must be between 256 and 1024 tokens (safety margin for 8192 limit)
    - Semantic boundaries are preserved where possible
    - Small chunks (<256) are merged with next chunk
    - Large chunks (>1024) trigger reprocessing with lower breakpoint
    """
    
    MIN_TOKENS = 256  # Reverted to 256
    MAX_TOKENS = 1024  # Reverted to 1024
    
    def merge_small_chunks(chunks: List[Document]) -> List[Document]:
        """Merge chunks smaller than MIN_TOKENS with next chunk"""
        if not chunks:
            return chunks
            
        merged = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue
                
            current_length = get_token_count(current_chunk.text)
            next_length = get_token_count(chunk.text)
            
            # If current chunk is too small and combining won't exceed max
            if current_length < MIN_TOKENS and (current_length + next_length) <= MAX_TOKENS:
                # Merge chunks
                current_chunk.text = current_chunk.text + " " + chunk.text
                # Update metadata to reflect merge
                current_chunk.metadata["chunk_id"] += f"_merged_{chunk.metadata['chunk_id']}"
            else:
                if current_length >= MIN_TOKENS:  # Only add if meets minimum size
                    merged.append(current_chunk)
                elif current_length > 0:  # Don't lose content, force include
                    logger.warning(f"Small chunk {current_length} tokens forced to include")
                    merged.append(current_chunk)
                current_chunk = chunk
        
        # Don't forget the last chunk
        if current_chunk:
            current_length = get_token_count(current_chunk.text)
            if current_length >= MIN_TOKENS or current_length > 0:  # Include any remaining content
                merged.append(current_chunk)
            
        return merged
    
    def split_large_chunk(text: str, metadata: Dict[str, Any], chunk_id: str) -> List[Document]:
        """Split a chunk that's too large using simple token splitting"""
        token_splitter = TokenTextSplitter(
            chunk_size=MAX_TOKENS - 50,  # Leave some margin
            chunk_overlap=20
        )
        
        temp_doc = Document(text=text, metadata=metadata)
        chunks = token_splitter.get_nodes_from_documents([temp_doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_id"] = f"{chunk_id}_split_{i}"
            chunk.metadata["text_type"] = "token_split"
        
        return chunks
    
    def process_with_breakpoint(text: str, breakpoint: int = 90) -> List[Document]:
        """Process text with given breakpoint, splitting large chunks as needed"""
        try:
            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=breakpoint,
                embed_model=Settings.embed_model
            )
            
            # Create initial document
            base_doc = Document(text=text, metadata=metadata)
            chunks = semantic_splitter.get_nodes_from_documents([base_doc])
            
            # Process each chunk for size
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_size = get_token_count(chunk.text)
                chunk.metadata["chunk_id"] = f"sem_{i}"
                chunk.metadata["text_type"] = "semantic"
                
                if chunk_size > MAX_TOKENS:
                    logger.info(f"Splitting large semantic chunk ({chunk_size} tokens)")
                    split_chunks = split_large_chunk(chunk.text, chunk.metadata, f"sem_{i}")
                    processed_chunks.extend(split_chunks)
                else:
                    processed_chunks.append(chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}, falling back to token splitting")
            # Fallback to simple token splitting
            token_splitter = TokenTextSplitter(
                chunk_size=MAX_TOKENS - 50,
                chunk_overlap=20
            )
            
            fallback_doc = Document(text=text, metadata=metadata)
            chunks = token_splitter.get_nodes_from_documents([fallback_doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(metadata)
                chunk.metadata["chunk_id"] = f"fallback_{i}"
                chunk.metadata["text_type"] = "token_fallback"
            
            return chunks
    
    try:
        # Step 1: Initial semantic chunking with size constraint handling
        semantic_chunks = process_with_breakpoint(text)
        logger.info(f"Created {len(semantic_chunks)} initial semantic chunks")
        
        # Step 2: Apply sentence windows to smaller chunks only
        sentence_window_parser = SentenceWindowNodeParser.from_defaults(
            window_size=2,  # Reduced from 3 to save tokens
            window_metadata_key="window",
            original_text_metadata_key="original_sentence"
        )
        
        all_chunks = []
        for chunk in semantic_chunks:
            chunk_size = get_token_count(chunk.text)
            
            # Only apply sentence windows to chunks that aren't too large
            if chunk_size <= MAX_TOKENS // 2:  # Only smaller chunks get sentence windows
                try:
                    chunk_doc = Document(text=chunk.text, metadata=chunk.metadata)
                    sentence_chunks = sentence_window_parser.get_nodes_from_documents([chunk_doc])
                    
                    for j, sentence_chunk in enumerate(sentence_chunks):
                        # Store window context as node attributes
                        window_text = sentence_chunk.metadata.get("window", "")
                        original_sentence = sentence_chunk.metadata.get("original_sentence", "")
                        
                        # Check window size before storing
                        if get_token_count(window_text) <= MAX_TOKENS:
                            setattr(sentence_chunk, '_window_context', window_text)
                        else:
                            # Truncate window if too large
                            tokens = encoding.encode(window_text)[:MAX_TOKENS]
                            setattr(sentence_chunk, '_window_context', encoding.decode(tokens))
                        
                        setattr(sentence_chunk, '_original_sentence', original_sentence)
                        
                        # Update metadata
                        sentence_chunk.metadata.update(chunk.metadata)
                        sentence_chunk.metadata["text_type"] = "sentence_window"
                        sentence_chunk.metadata["chunk_id"] = f"{chunk.metadata['chunk_id']}_sent_{j}"
                        
                        # Remove large fields from metadata
                        sentence_chunk.metadata.pop("window", None)
                        sentence_chunk.metadata.pop("original_sentence", None)
                        
                        # Final size check
                        final_size = get_token_count(sentence_chunk.text)
                        if final_size <= MAX_TOKENS:
                            all_chunks.append(sentence_chunk)
                        else:
                            logger.warning(f"Skipping oversized sentence chunk: {final_size} tokens")
                
                except Exception as e:
                    logger.warning(f"Sentence window failed for chunk, using original: {e}")
                    all_chunks.append(chunk)
            else:
                # Large chunks skip sentence windows
                all_chunks.append(chunk)
        
        # Step 3: Final merge pass for small chunks
        final_chunks = merge_small_chunks(all_chunks)
        
        # Step 4: Final validation
        validated_chunks = []
        for chunk in final_chunks:
            size = get_token_count(chunk.text)
            if size <= MAX_TOKENS:
                validated_chunks.append(chunk)
            else:
                logger.error(f"Final chunk still too large: {size} tokens, splitting...")
                split_chunks = split_large_chunk(chunk.text, chunk.metadata, chunk.metadata.get('chunk_id', 'oversized'))
                validated_chunks.extend(split_chunks)
        
        # Log final statistics
        sizes = [get_token_count(chunk.text) for chunk in validated_chunks]
        logger.info(f"Created {len(validated_chunks)} final chunks. Size range: {min(sizes)}-{max(sizes)} tokens")
        
        return validated_chunks
    
    except Exception as e:
        logger.error(f"All chunking failed: {e}")
        # Emergency fallback - very conservative token splitting
        token_splitter = TokenTextSplitter(
            chunk_size=500,  # Very conservative
            chunk_overlap=20
        )
        
        emergency_doc = Document(text=text, metadata=metadata)
        emergency_chunks = token_splitter.get_nodes_from_documents([emergency_doc])
        
        for i, chunk in enumerate(emergency_chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["text_type"] = "emergency_fallback"
            chunk.metadata["chunk_id"] = f"emergency_{i}"
        
        logger.info(f"Created {len(emergency_chunks)} emergency fallback chunks")
        return emergency_chunks

def create_unified_citation(meta: Dict[str, Any]) -> str:
    """
    Create citations using the simplified paragraph-based approach.
    Each chunk is one paragraph, so citations are direct and accurate.
    """
    # Use the new simplified citation function
    return create_simplified_citation(meta)

def format_source_citation(meta: Dict[str, Any]) -> str:
    """
    Format metadata into a clean source citation for UI display.
    Uses the simplified citation approach.
    """
    return create_simplified_citation(meta)

class NumberedCitationReplacer:
    """Replaces numbered citations [1], [2], etc. with actual metadata-based citations or plain inline citations."""
    
    def __init__(self, source_nodes, use_full_citations=True):
        self.source_nodes = source_nodes
        self.citation_map = {}
        self.use_full_citations = use_full_citations
        
        # Create citation map from source nodes
        for i, node in enumerate(source_nodes, 1):
            meta = getattr(node.node, "metadata", {})
            if self.use_full_citations:
                citation = create_simplified_citation(meta)
                # Wrap full citations in brackets with gray italic styling
                citation = f'<span style="color: #666666; font-style: italic;">({citation})</span>'
            else:
                # Use numbered citations with gray italic styling
                citation = f'<span style="color: #666666; font-style: italic;">[{i}]</span>'
            self.citation_map[i] = citation
    
    def replace_citations(self, response_text: str) -> str:
        """
        Replace numbered citations [1], [2], etc. with actual metadata citations or plain inline citations.
        """
        if not self.source_nodes:
            return response_text
        
        # If using numbered citations, replace with styled numbered citations
        if not self.use_full_citations:
            # For numbered citations mode, replace with styled citations from citation map
            modified_text = response_text
            
            # First, replace any existing HTML anchor links with plain citations
            html_pattern = r'<a href="#source-(\d+)">\[(\d+)\]</a>'
            modified_text = re.sub(html_pattern, r'[\2]', modified_text)
            
            # Then replace numbered citations with styled versions from citation map
            citation_pattern = r'\[(\d+)\]'
            matches = list(re.finditer(citation_pattern, modified_text))
            for match in reversed(matches):
                citation_num = int(match.group(1))
                if citation_num in self.citation_map:
                    start, end = match.span()
                    modified_text = (
                        modified_text[:start] + 
                        self.citation_map[citation_num] + 
                        modified_text[end:]
                    )
            return modified_text
        
        # Replace numbered citations with actual metadata
        # Also handle any existing HTML anchor links and replace them with full citations
        modified_text = response_text
        
        # First, replace any existing HTML anchor links with just the citation number
        html_pattern = r'<a href="#source-(\d+)">\[(\d+)\]</a>'
        modified_text = re.sub(html_pattern, r'[\2]', modified_text)
        
        # Then replace numbered citations with full citations
        citation_pattern = r'\[(\d+)\]'
        matches = list(re.finditer(citation_pattern, modified_text))
        for match in reversed(matches):
            citation_num = int(match.group(1))
            if citation_num in self.citation_map:
                start, end = match.span()
                modified_text = (
                    modified_text[:start] + 
                    " " + self.citation_map[citation_num] + 
                    modified_text[end:]
                )
        return modified_text

@st.cache_resource(show_spinner=False)
def load_and_index_documents():
    """Load documents and create searchable index with simplified paragraph-based chunking"""
    
    persist_dir = "storage_v3_paragraph"  # New storage directory for paragraph-based index
    
    # Try to load existing index
    if os.path.exists(persist_dir):
        try:
            # Show loading indicator for existing index
            with st.spinner("Loading existing index from storage..."):
                logger.info("Loading existing paragraph-based index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)
                logger.info("Successfully loaded existing paragraph-based index")
                st.success("Index loaded successfully!")
                return index
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            st.warning("Failed to load existing index. Building new one...")
            logger.info("Building new paragraph-based index...")
    
    # Show progress for new index creation
    progress_container = st.container()
    with progress_container:
        st.markdown("### Building New Index")
        progress_bar = st.progress(0)
        status_text = st.empty()
        phase_text = st.empty()
    
    # Phase 1: Load TOC (10%)
    phase_text.text("Phase 1/5: Loading document structure...")
    status_text.text("Reading table of contents...")
    with open("toc.json", "r", encoding="utf-8") as f:
        toc = json.load(f)
    progress_bar.progress(0.1)
    
    # Phase 2: Configure models (20%)
    phase_text.text("Phase 2/5: Configuring AI models...")
    status_text.text("Setting up language model and embeddings...")
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.05,  # Very low temperature for maximum accuracy
        system_prompt="""You are an expert on Building Regulations for England. 
        Answer questions with 100% accuracy based strictly on the provided Approved Documents.
        
        CITATION REQUIREMENTS:
        - Include numbered citations [1], [2], [3] etc. after factual statements from the reference material
        - Citations must appear immediately after technical facts, requirements, and specifications
        - Each source number corresponds to the source order in the context
        - Example: "Buildings must have fire resistance [1]. The minimum rating is 30 minutes [2]."
        - Do NOT cite conversational responses, acknowledgments, or general statements
        - Only cite actual regulatory content from the provided sources"""
    )
    
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=20  # Can increase since chunks are more uniform
    )
    progress_bar.progress(0.2)
    
    # Phase 3: Extract paragraphs (60%)
    phase_text.text("Phase 3/5: Extracting paragraphs from documents...")
    status_text.text("Processing PDF pages and extracting regulatory paragraphs...")
    pdf_path = "data/The_Merged_Approved_Documents_Oct24.pdf"
    
    # Use the new simplified chunking
    all_chunks = create_paragraph_based_chunks(pdf_path, toc)
    progress_bar.progress(0.8)
    
    # Phase 4: Build index (80%)
    phase_text.text("Phase 4/5: Building searchable index...")
    status_text.text(f"Creating vector index from {len(all_chunks)} paragraphs...")
    logger.info(f"Creating index from {len(all_chunks)} paragraph chunks...")
    index = VectorStoreIndex(all_chunks)
    progress_bar.progress(0.9)
    
    # Phase 5: Persist index (100%)
    phase_text.text("Phase 5/5: Saving index to storage...")
    status_text.text("Persisting index for future use...")
    logger.info("Persisting paragraph-based index to storage...")
    index.storage_context.persist(persist_dir=persist_dir)
    progress_bar.progress(1.0)
    
    # Clear progress indicators and show success
    progress_container.empty()
    st.success("Index ready! A.D.R.E is now available for queries.")
    
    logger.info("Simplified paragraph-based indexing complete!")
    return index

# Load index
index = load_and_index_documents()

# Chat interface
if st.session_state.selected_mode_key == "reference":
    prompt = st.chat_input("Ask a question about Building Regulations")
    if prompt:
        st.session_state.messages_v2.append({"role": "user", "content": prompt, "sources": None, "citation_map": None})
        st.session_state.generating_reference = True
    
    # Find the index of the last user message
    last_user_idx = None
    for i in range(len(st.session_state.messages_v2) - 1, -1, -1):
        if st.session_state.messages_v2[i]["role"] == "user":
            last_user_idx = i
            break
    
    # Render logic
    if st.session_state.generating_reference and last_user_idx is not None:
        # Render all messages up to and including the last user message
        messages_to_render = st.session_state.messages_v2[:last_user_idx+1]
    else:
        messages_to_render = st.session_state.messages_v2
    
    for message in messages_to_render:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message.get("sources"):
                st.markdown("### Reference")
                for i, source in enumerate(message["sources"], 1):
                    meta = source["meta"]
                    citation = source["citation"]
                    text_preview = source["text_preview"]
                    with st.expander(f"Source {i} ({citation})"):
                        st.text(text_preview)
                        pdf_base_url = "https://assets.publishing.service.gov.uk/media/6717d29438149ce9d09e3862/The_Merged_Approved_Documents_Oct24.pdf"
                        page_num = meta.get('pdf_page', 1)
                        pdf_link = f"[Open PDF to page {page_num}]({pdf_base_url}#page={page_num})"
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if meta.get('document_title', 'Unknown') != 'Unknown':
                                st.text(f"Document: {meta['document_title']}")
                            if meta.get('section', 'Unknown') != 'Unknown':
                                st.text(f"Section: {meta['section']}")
                            if meta.get('paragraph'):
                                st.text(f"Paragraph: {meta['paragraph']}")
                        with col2:
                            if meta.get('internal_page'):
                                st.text(f"Page: {meta['internal_page']}")
                            if meta.get('regulation_ref'):
                                st.text(f"Part: {meta['regulation_ref']}")
                            if meta.get('requirement_ref'):
                                st.text(f"Requirement: {meta['requirement_ref']}")
                        with col3:
                            if meta.get('sub_items'):
                                st.text(f"Sub-items: {', '.join(meta['sub_items'])}")
                            if meta.get('diagram'):
                                st.text(f"Contains: {meta['diagram']}")
    
    # Show spinner only if last message is from user and generating
    if st.session_state.generating_reference and last_user_idx is not None and last_user_idx == len(st.session_state.messages_v2) - 1:
        with st.chat_message("assistant"):
            with st.spinner("Searching regulatory paragraphs..."):
                # Check if query contains a document reference
                target_document = extract_document_reference_from_query(prompt)
                
                if target_document:
                    # Use document-filtered query engine
                    current_engine = create_document_filtered_query_engine(index, target_document)
                    st.info(f"Filtering results to Document {target_document}")
                else:
                    # Use standard query engine
                    current_engine = get_query_engine(index, similarity_top_k, citation_chunk_size, use_window_context)
                
                response = current_engine.query(prompt)
            cited_response = response.response  # Default to original response
            sources_list = []
            if hasattr(response, "source_nodes") and response.source_nodes:
                citation_replacer = NumberedCitationReplacer(response.source_nodes, st.session_state.use_full_citations)
                cited_response = citation_replacer.replace_citations(response.response)
                for i, node in enumerate(response.source_nodes, 1):
                    meta = getattr(node.node, "metadata", {})
                    citation = format_source_citation(meta)
                    text_preview = node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text
                    sources_list.append({"meta": meta, "citation": citation, "text_preview": text_preview})
            st.markdown(cited_response, unsafe_allow_html=True)
            # Show Reference list only when not in full citation mode
            if sources_list and not st.session_state.use_full_citations:
                st.markdown("### Reference")
                for i, source in enumerate(sources_list, 1):
                    # Add anchor div for scroll target
                    st.markdown(f'<div id="source-{i}"></div>', unsafe_allow_html=True)
                    meta = source["meta"]
                    citation = source["citation"]
                    text_preview = source["text_preview"]
                    with st.expander(f"Source {i} ({citation})"):
                        st.text(text_preview)
                        pdf_base_url = "https://assets.publishing.service.gov.uk/media/6717d29438149ce9d09e3862/The_Merged_Approved_Documents_Oct24.pdf"
                        page_num = meta.get('pdf_page', 1)
                        pdf_link = f"[Open PDF to page {page_num}]({pdf_base_url}#page={page_num})"
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if meta.get('document_title', 'Unknown') != 'Unknown':
                                st.text(f"Document: {meta['document_title']}")
                            if meta.get('section', 'Unknown') != 'Unknown':
                                st.text(f"Section: {meta['section']}")
                            if meta.get('paragraph'):
                                st.text(f"Paragraph: {meta['paragraph']}")
                        with col2:
                            if meta.get('internal_page'):
                                st.text(f"Page: {meta['internal_page']}")
                            if meta.get('regulation_ref'):
                                st.text(f"Part: {meta['regulation_ref']}")
                            if meta.get('requirement_ref'):
                                st.text(f"Requirement: {meta['requirement_ref']}")
                        with col3:
                            if meta.get('sub_items'):
                                st.text(f"Sub-items: {', '.join(meta['sub_items'])}")
                            if meta.get('diagram'):
                                st.text(f"Contains: {meta['diagram']}")
            st.session_state.messages_v2.append({"role": "assistant", "content": cited_response, "sources": sources_list, "citation_map": None})
            st.session_state.generating_reference = False  # Clear the flag
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.messages_v2 = [
                {"role": "assistant", "content": "A.D.R.E Ready."}
            ]
            st.session_state.generating_reference = False
            st.rerun()
else:
    prompt = st.chat_input("Chat with ADRE about Building Regulations")
    if prompt:
        st.session_state.chat_mode_history.append({"role": "user", "content": prompt})
        st.session_state.generating_chat = True
    
    # Find the index of the last user message
    last_user_idx = None
    for i in range(len(st.session_state.chat_mode_history) - 1, -1, -1):
        if st.session_state.chat_mode_history[i]["role"] == "user":
            last_user_idx = i
            break
    
    # Render logic
    if st.session_state.generating_chat and last_user_idx is not None:
        # Render all messages up to and including the last user message
        messages_to_render = st.session_state.chat_mode_history[:last_user_idx+1]
    else:
        messages_to_render = st.session_state.chat_mode_history
    
    for message in messages_to_render:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Show spinner only if last message is from user and generating
    if st.session_state.generating_chat and last_user_idx is not None and last_user_idx == len(st.session_state.chat_mode_history) - 1:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the current chat mode from session state
                chat_engine = get_chat_engine(index, st.session_state.current_chat_mode)
                response = chat_engine.chat(prompt)
                formatted_response = response.response
                sources_list = []
                if hasattr(response, "source_nodes") and response.source_nodes:
                    citation_replacer = NumberedCitationReplacer(response.source_nodes, st.session_state.use_full_citations)
                    formatted_response = citation_replacer.replace_citations(response.response)
                    for i, node in enumerate(response.source_nodes, 1):
                        meta = getattr(node.node, "metadata", {})
                        citation = format_source_citation(meta)
                        text_preview = node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text
                        sources_list.append({"meta": meta, "citation": citation, "text_preview": text_preview})
                st.markdown(formatted_response, unsafe_allow_html=True)
                # Show Reference list only when not in full citation mode and there are sources
                if sources_list and not st.session_state.use_full_citations:
                    st.markdown("### Reference")
                    for i, source in enumerate(sources_list, 1):
                        st.markdown(f'<div id="source-{i}"></div>', unsafe_allow_html=True)
                        meta = source["meta"]
                        citation = source["citation"]
                        text_preview = source["text_preview"]
                        with st.expander(f"Source {i} ({citation})"):
                            st.text(text_preview)
                            pdf_base_url = "https://assets.publishing.service.gov.uk/media/6717d29438149ce9d09e3862/The_Merged_Approved_Documents_Oct24.pdf"
                            page_num = meta.get('pdf_page', 1)
                            pdf_link = f"[Open PDF to page {page_num}]({pdf_base_url}#page={page_num})"
                            st.markdown(pdf_link, unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if meta.get('document_title', 'Unknown') != 'Unknown':
                                    st.text(f"Document: {meta['document_title']}")
                                if meta.get('section', 'Unknown') != 'Unknown':
                                    st.text(f"Section: {meta['section']}")
                                if meta.get('paragraph'):
                                    st.text(f"Paragraph: {meta['paragraph']}")
                            with col2:
                                if meta.get('internal_page'):
                                    st.text(f"Page: {meta['internal_page']}")
                                if meta.get('regulation_ref'):
                                    st.text(f"Part: {meta['regulation_ref']}")
                                if meta.get('requirement_ref'):
                                    st.text(f"Requirement: {meta['requirement_ref']}")
                            with col3:
                                if meta.get('sub_items'):
                                    st.text(f"Sub-items: {', '.join(meta['sub_items'])}")
                                if meta.get('diagram'):
                                    st.text(f"Contains: {meta['diagram']}")
                st.session_state.chat_mode_history.append({"role": "assistant", "content": formatted_response})
                st.session_state.generating_chat = False  # Clear the flag
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear History", type="secondary", use_container_width=True, key="reset_chat_mode"):
            st.session_state.chat_mode_history = [
                {"role": "assistant", "content": "A.D.R.E Ready."}
            ]
            st.session_state.generating_chat = False
            st.rerun() 