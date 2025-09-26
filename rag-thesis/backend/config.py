import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
    AZURE_SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "case-matthijs-index")

    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    COHERE_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-3.5")

    # Blob storage for clickable sources
    AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    AZURE_STORAGE_BLOB_ENDPOINT = os.environ.get(
        "AZURE_STORAGE_BLOB_ENDPOINT",
        f"https://{os.environ.get('AZURE_STORAGE_ACCOUNT_NAME', '')}.blob.core.windows.net" if os.environ.get('AZURE_STORAGE_ACCOUNT_NAME') else None,
    )
    AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER")
    THESIS_BLOB_NAME = os.environ.get("THESIS_BLOB_NAME")
    AZURE_BLOB_SAS_TOKEN = os.environ.get("AZURE_BLOB_SAS_TOKEN") 
    AZURE_BLOB_SAS_TTL_SECONDS = int(os.environ.get("AZURE_BLOB_SAS_TTL_SECONDS", "900"))

settings = Settings()
