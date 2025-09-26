import os
import json
import uuid
import hashlib
import re
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = os.environ.get("THESIS_PDF", "MSc_Thesis_Econometrics_final_version.pdf")
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "case-matthijs-index")

# Version for hash computation - bump when settings change
HASH_SPEC_VERSION = "1.0"

def canonicalize_value(value: Any) -> str:
    """Canonicalize a value for consistent hashing."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Normalize whitespace and Unicode
        normalized = re.sub(r'\s+', ' ', value.strip())
        return normalized
    elif isinstance(value, dict):
        # Sort keys for deterministic ordering
        sorted_items = sorted(value.items())
        return json.dumps({k: canonicalize_value(v) for k, v in sorted_items}, sort_keys=True, ensure_ascii=False)
    elif isinstance(value, list):
        return json.dumps([canonicalize_value(v) for v in value], sort_keys=True, ensure_ascii=False)
    else:
        return str(value)

def compute_content_hash(text: str, summary: str, key_words: str, page_number: int | None, source: str) -> str:
    """
    Compute a deterministic hash for content that should trigger re-indexing.
    
    Acceptance criteria:
    - Deterministic: same inputs → same hash, regardless of dict key order or whitespace
    - Canonicalized: normalized paths, newlines, Unicode, numbers, and booleans
    - Versioned: bump spec_version when summarizer/embedding/splitter settings change
    - Minimal but complete: include only fields that should trigger re-indexing
    - Opaque: use SHA-256 hex; never tokenize/search this field
    """
    # Define the minimal set of fields that should trigger re-indexing
    hash_data = {
        "spec_version": HASH_SPEC_VERSION,
        "text": text,
        "summary": summary,
        "key_words": key_words,
        "page_number": page_number,
        "source": source,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "encoding": "o200k_base"
    }
    
    # Canonicalize the entire data structure
    canonical_data = canonicalize_value(hash_data)
    
    # Compute SHA-256 hash
    return hashlib.sha256(canonical_data.encode('utf-8')).hexdigest()

async def generate_chunk_summary(text: str) -> str:
    """Generate a concise summary of a text chunk using OpenAI."""
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at creating concise, informative summaries. Create a brief summary (1-2 sentences) that captures the key concepts and main points of the given text. Focus on the most important information that would be useful for retrieval and question-answering."),
        ("human", "Text to summarize:\n\n{text}\n\nSummary:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"text": text})

async def generate_key_words(text: str) -> str:
    """Extract key words and phrases from a text chunk using OpenAI."""
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0.1,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting key words and phrases. Extract the most important keywords, technical terms, concepts, and phrases from the given text. Return them as a comma-separated list. Focus on terms that would be useful for search and retrieval. Keep it concise but comprehensive."),
        ("human", "Text to extract keywords from:\n\n{text}\n\nKey words:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"text": text})
 
def build_vectorstore():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    vector_store = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_ADMIN_KEY"],
        index_name=INDEX_NAME,
        embedding_function=embeddings.embed_query,
        additional_search_client_options={"retry_total": 4},
    )
    return vector_store

async def process_chunk_async(chunk, chunk_index: int, document_id: str, pdf_basename: str) -> dict:
    """Process a single chunk asynchronously to generate summary and key words."""
    print(f"Processing chunk {chunk_index + 1}...")
    
    # Generate summary and key words concurrently
    summary_task = generate_chunk_summary(chunk.page_content)
    key_words_task = generate_key_words(chunk.page_content)
    
    # Wait for both tasks to complete
    summary, key_words_text = await asyncio.gather(summary_task, key_words_task)
    
    # Convert key words to array of strings (split by comma and clean)
    key_words = [kw.strip() for kw in key_words_text.split(',') if kw.strip()]
    
    # Generate unique chunk ID and timestamp
    chunk_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    
    src = chunk.metadata.get("source") or pdf_basename
    page_num = chunk.metadata.get("page_number")
    if page_num is None:
        page_num = chunk.metadata.get("page")
    # Ensure page_num is always an int or None
    page_num = int(page_num) if page_num is not None else None
    
    # Compute content hash
    content_hash = compute_content_hash(chunk.page_content, summary, key_words_text, page_num, src)
    
    return {
        "source": src, 
        "page_number": page_num, 
        "summary": summary, 
        "chunk_id": chunk_id, 
        "created_at": created_at,
        "key_words": key_words,
        "full_text": chunk.page_content,
        "document_id": document_id,
        "meta_hash": content_hash,
        "chunk_index": chunk_index
    }

async def main_async():
    assert os.path.exists(PDF_PATH), f"PDF not found: {PDF_PATH}"

    # 1) Load PDF (returns per-page documents incl. page metadata)
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # 2) Split into token-based chunks
    splitter = TokenTextSplitter(
        chunk_size=512, chunk_overlap=50, encoding_name="o200k_base" 
    )
    chunks = splitter.split_documents(pages)

    # Generate document ID for the entire PDF
    document_id = str(uuid.uuid4())
    pdf_basename = os.path.basename(PDF_PATH)
    
    print("Generating summaries and key words for chunks asynchronously...")
    
    # Process all chunks concurrently with a reasonable concurrency limit
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    
    async def process_with_semaphore(chunk, index):
        async with semaphore:
            return await process_chunk_async(chunk, index, document_id, pdf_basename)
    
    # Create tasks for all chunks
    tasks = [process_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)]
    
    # Wait for all tasks to complete
    chunk_metadata_list = await asyncio.gather(*tasks)
    
    # Update chunk metadata
    for chunk, metadata in zip(chunks, chunk_metadata_list):
        chunk.metadata = metadata

    # 3) Compute embeddings and upload via Azure Search SDK
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    # Generate embeddings for both content and summaries
    print("Generating embeddings...")
    content_texts = [d.page_content for d in chunks]
    summary_texts = [d.metadata.get("summary", "") for d in chunks]
    
    content_vectors = embeddings.embed_documents(content_texts)
    summary_vectors = embeddings.embed_documents(summary_texts)

    client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )

    documents = []
    for d, content_vec, summary_vec in zip(chunks, content_vectors, summary_vectors):
        meta_obj = {
            "source": d.metadata.get("source"), 
            "page_number": d.metadata.get("page_number"),
            "summary": d.metadata.get("summary"),
            "chunk_id": d.metadata.get("chunk_id"),
            "chunk_index": d.metadata.get("chunk_index"),
            "created_at": d.metadata.get("created_at"),
            "key_words": d.metadata.get("key_words"),
            "full_text": d.metadata.get("full_text"),
            "document_id": d.metadata.get("document_id"),
            "meta_hash": d.metadata.get("meta_hash")
        }
        documents.append({
            "id": str(uuid.uuid4()),
            "content": d.page_content,
            "content_vector": content_vec,
            "summary_vector": summary_vec,
            "metadata": meta_obj,
            "metadata_text": json.dumps(meta_obj, ensure_ascii=False, default=str),
        })

    results = client.upload_documents(documents=documents)
    succeeded = sum(1 for r in results if getattr(r, "succeeded", False))
    print(f"Indexed {succeeded}/{len(documents)} chunks into '{INDEX_NAME}'.")

def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
