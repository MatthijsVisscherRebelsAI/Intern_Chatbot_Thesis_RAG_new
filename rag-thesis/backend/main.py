from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langsmith.middleware import TracingMiddleware  
from .rag_chain import build_chain
from .config import settings
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import json
import asyncio
from langsmith import Client
from datetime import datetime, timedelta, timezone
import os
try:
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
except Exception:
    generate_blob_sas = None  # type: ignore
    BlobSasPermissions = None  # type: ignore

app = FastAPI(title="RAG Thesis API")
app.add_middleware(TracingMiddleware)  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve the frontend directory relative to this file
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# Serve static assets (e.g., app.js) from /static
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

chain, retriever, rerank = build_chain()
client = Client()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: AskRequest):
    answer = chain.invoke({"question": payload.question}, config={"tags": ["http", "ask", "proj:thesis", "stage:dev"], "metadata": {"route": "/ask"}})
    return {"answer": answer}

@app.get("/stream")
async def stream(question: str):
    async def event_generator():
        citations = []
        # Buffer generated text to detect "don't know" style answers
        answer_buffer = []

        # Keepalive pings
        async def keepalive():
            try:
                while True:
                    # Comment line per SSE spec
                    yield ": ping\n\n"
                    await asyncio.sleep(20)
            except asyncio.CancelledError:
                return

        keepalive_task = asyncio.create_task(asyncio.sleep(0))
        try:
            # Start keepalive in the background via manual interleave
            last_ping = asyncio.get_event_loop().time()
            async for chunk in chain.astream({"question": question}, config={"tags": ["http", "stream", "proj:thesis", "stage:dev"], "metadata": {"route": "/stream"}}):
                if chunk:
                    yield "event: chunk\n" + "data: " + json.dumps({"delta": chunk}) + "\n\n"
                    # Accumulate a bounded buffer for simple detection
                    answer_buffer.append(str(chunk))
                    if len(answer_buffer) > 2000:
                        # keep last ~2k chars
                        joined = "".join(answer_buffer)
                        answer_buffer = [joined[-2000:]]
                # interleave ping every ~20s
                now = asyncio.get_event_loop().time()
                if now - last_ping > 20:
                    yield ": ping\n\n"
                    last_ping = now
        finally:
            if not keepalive_task.done():
                keepalive_task.cancel()

        # Final event: gather citations from the reranked set actually used for generation
        try:
            # If the model effectively said it doesn't know, suppress citations
            text = ("".join(answer_buffer)).strip().lower()
            unknown_markers = [
                "i don't know",
                "i do not know",
                "unknown",
                "not in the context",
                "not found in the context",
                "insufficient information",
                "no information in the context",
                "cannot answer based on the provided context",
            ]
            should_suppress = any(m in text for m in unknown_markers)

            try:
                docs = rerank(question)
            except Exception:
                docs = retriever.invoke(question)
            # Normalize source label and deduplicate by page
            normalized = []
            seen_pages = set()
            for d in docs:
                # Normalize page metadata from either 'page_number' or 'page'
                page = d.metadata.get("page_number", d.metadata.get("page"))
                if page in seen_pages:
                    continue
                seen_pages.add(page)
                raw_source = d.metadata.get("source", "thesis.pdf")
                # Prefer configured blob name; else basename of the source path
                label = settings.THESIS_BLOB_NAME or os.path.basename(str(raw_source)) or "thesis.pdf"
                normalized.append({"source": label, "page": page})
            citations = [] if should_suppress else normalized
        except Exception:
            citations = []

        # Prefer generating a short-lived SAS for private blob access
        try:
            if (
                settings.AZURE_STORAGE_ACCOUNT_NAME
                and settings.AZURE_STORAGE_ACCOUNT_KEY
                and settings.AZURE_STORAGE_CONTAINER
                and settings.THESIS_BLOB_NAME
                and generate_blob_sas is not None
                and BlobSasPermissions is not None
            ):
                expiry = datetime.now(timezone.utc) + timedelta(seconds=settings.AZURE_BLOB_SAS_TTL_SECONDS)
                sas = generate_blob_sas(
                    account_name=settings.AZURE_STORAGE_ACCOUNT_NAME,
                    container_name=settings.AZURE_STORAGE_CONTAINER,
                    blob_name=settings.THESIS_BLOB_NAME,
                    account_key=settings.AZURE_STORAGE_ACCOUNT_KEY,
                    permission=BlobSasPermissions(read=True),
                    expiry=expiry,
                )
                base = f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{settings.AZURE_STORAGE_CONTAINER}/{settings.THESIS_BLOB_NAME}?{sas}"
                citations = [
                    {**c, "url": f"{base}#page={c['page']}" if (c.get("page") is not None) else base}
                    for c in citations
                ]
            # Fallback to pre-provided SAS or public container endpoint
            elif settings.AZURE_STORAGE_BLOB_ENDPOINT and settings.AZURE_STORAGE_CONTAINER and settings.THESIS_BLOB_NAME:
                base_no_sas = f"{settings.AZURE_STORAGE_BLOB_ENDPOINT}/{settings.AZURE_STORAGE_CONTAINER}/{settings.THESIS_BLOB_NAME}"
                sas_token = settings.AZURE_BLOB_SAS_TOKEN or ""
                if sas_token and not sas_token.startswith("?"):
                    sas_token = "?" + sas_token
                base = f"{base_no_sas}{sas_token}"
                citations = [
                    {**c, "url": f"{base}#page={c['page']}" if (c.get("page") is not None) else base}
                    for c in citations
                ]
        except Exception:
            # If SAS generation fails, we still send plain labels without URLs
            pass
        yield "event: done\n" + "data: " + json.dumps({"citations": citations}) + "\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(index_path)

