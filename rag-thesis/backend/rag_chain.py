from __future__ import annotations
from typing import List, Dict, Any
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from .config import settings
from langsmith.run_helpers import traceable, get_current_run_tree
try:
    from langchain_cohere import CohereRerank
except Exception:
    CohereRerank = None  # type: ignore

# LangSmith tracing constants
RUN_NAME = "rag_chain"
TAGS = ["proj:thesis", "stage:dev", "retriever:hybrid-rrf", "gen:responses"]
VERSION = "2025-09-17"
RETRIEVER_K = 8  # number of candidate chunks to retrieve; keep K/2 after rerank
logger = logging.getLogger("rag.rerank")

def run_meta(question: str, retrieved_docs, reranked_docs):
    hits = []
    for d in reranked_docs or []:
        hits.append({
            "id": d.metadata.get("id"),
            "page": d.metadata.get("page"),
            "score": d.metadata.get("score"),
        })
    ctx_pages = sorted({d.metadata.get("page") for d in reranked_docs or [] if "page" in d.metadata})
    unique_docs = len({d.metadata.get("doc_id") for d in reranked_docs or []})
    return {
        "question": question,
        "version": VERSION,
        "retriever": {
            "k_bm25": 20,
            "k_vect": 20,
            "k_fused": len(reranked_docs or []),
            "rrf_k": 60,
            "hits": hits,
        },
        "ctx_stats": {
            "unique_docs": unique_docs,
            "pages": ctx_pages,
        },
    }

def _format_docs(docs):
    # Deduplicate by page so we don't repeat similar content
    seen_pages = set()
    unique_docs = []
    for d in docs:
        page = d.metadata.get("page_number", d.metadata.get("page"))
        if page in seen_pages:
            continue
        seen_pages.add(page)
        unique_docs.append(d)

    lines = []
    for d in unique_docs:
        page = d.metadata.get("page_number", d.metadata.get("page"))
        lines.append(f"[p.{page}] {d.page_content}")
    context = "\n\n".join(lines[:10])  # cap defensively
    sources = [
        {"source": d.metadata.get("source", "thesis.pdf"), "page": d.metadata.get("page_number", d.metadata.get("page"))}
        for d in unique_docs
    ]
    return context, sources

def build_chain(index_name: str | None = None):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    vector_store = AzureSearch(
        azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
        azure_search_key=settings.AZURE_SEARCH_ADMIN_KEY,
        index_name=index_name or settings.AZURE_SEARCH_INDEX,
        embedding_function=embeddings.embed_query,
        metadata_key="metadata_text",
        additional_search_client_options={"timeout": 30, "retry_total": 2},
    )

    # Retrieve K candidates (pure vector similarity for lower latency)
    retriever = vector_store.as_retriever(search_type="similarity", k=RETRIEVER_K)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant that must answer strictly using the provided thesis context."
             "If the information is not in the context, clearly say you don't know."
             "Provide answers in the style of the Journal of Econometrics, make it consisely."
             "Do not include a 'Sources' section; the application will display sources separately."),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely and precisely."),
        ]
    )

    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0.1,
        streaming=True,
    )

    @traceable(name="retrieve_and_rerank", tags=TAGS)
    def retrieve_and_rerank(q: str):
        candidates = retriever.invoke(q)
        top_m = max(1, min(len(candidates), RETRIEVER_K // 2))
        # Optional re-ranking with Cohere cross-encoder for higher precision
        try:
            if settings.COHERE_API_KEY and CohereRerank is not None and candidates:
                reranker = CohereRerank(model=settings.COHERE_RERANK_MODEL, cohere_api_key=settings.COHERE_API_KEY)
                logger.info("Cohere rerank active: model=%s K=%d keep=%d", settings.COHERE_RERANK_MODEL, len(candidates), top_m)
                reranked = reranker.compress_documents(candidates, query=q, top_n=top_m)
                return reranked
        except Exception:
            # If reranking fails for any reason, fall back gracefully
            pass
        return candidates[:top_m]

    def prep(inputs: Dict[str, Any]):
        # Allow precomputed docs to avoid double-retrieval from the endpoint
        docs = inputs.get("pre_docs") or retrieve_and_rerank(inputs["question"])
        context, sources = _format_docs(docs)
        # Log params/metadata inside this span if available
        try:
            rt = get_current_run_tree()
            if rt is not None:
                rt.log_params({"question": inputs["question"], "version": VERSION})
        except Exception:
            pass
        return {"question": inputs["question"], "context": context, "sources": sources}

    prep_lambda = RunnableLambda(prep).with_config(run_name="retrieve_and_prepare")

    chain = (
        RunnablePassthrough.assign(**{"question": lambda x: x["question"]}).with_config(run_name="passthrough_question")
        | prep_lambda
        | (prompt | llm | StrOutputParser()).with_config(run_name="prompt_generate_parse")
    ).with_config(run_name=RUN_NAME)

    # Also return the rerank function so callers can reuse the exact doc set for UI citations
    return chain, retriever, retrieve_and_rerank

