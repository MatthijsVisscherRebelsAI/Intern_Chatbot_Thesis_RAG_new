import os
import json
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

PDF_PATH = os.environ.get("THESIS_PDF", "MSc_Thesis_Econometrics_final_version.pdf")
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "case-matthijs-index")
 
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

def main():
    assert os.path.exists(PDF_PATH), f"PDF not found: {PDF_PATH}"

    # 1) Load PDF (returns per-page documents incl. page metadata)
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # 2) Split into token-based chunks
    splitter = TokenTextSplitter(
        chunk_size=512, chunk_overlap=50, encoding_name="o200k_base" 
    )
    chunks = splitter.split_documents(pages)

    for d in chunks:
        src = d.metadata.get("source") or os.path.basename(PDF_PATH)
        # Portal schema: metadata is ComplexType with subfields { source: String, page_number: Int32 }
        page_num = d.metadata.get("page_number")
        if page_num is None:
            page_num = d.metadata.get("page")
        # Coerce to int defensively
        try:
            page_num = int(page_num) if page_num is not None else None
        except Exception:
            page_num = None
        d.metadata = {"source": src, "page_number": page_num}

    # 3) Compute embeddings and upload via Azure Search SDK
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    texts = [d.page_content for d in chunks]
    vectors = embeddings.embed_documents(texts)

    client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )

    documents = []
    for d, vec in zip(chunks, vectors):
        meta_obj = {"source": d.metadata.get("source"), "page_number": d.metadata.get("page_number")}
        documents.append({
            "id": str(uuid.uuid4()),
            "content": d.page_content,
            "content_vector": vec,
            # ComplexType for portal/filters
            "metadata": meta_obj,
            # String duplication for LangChain retriever
            "metadata_text": json.dumps(meta_obj, ensure_ascii=False),
        })

    results = client.upload_documents(documents=documents)
    succeeded = sum(1 for r in results if getattr(r, "succeeded", False))
    print(f"Indexed {succeeded}/{len(documents)} chunks into '{INDEX_NAME}'.")

if __name__ == "__main__":
    main()
