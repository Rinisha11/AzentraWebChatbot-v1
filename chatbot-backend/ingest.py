# ingest.py - Upgraded for multi-tenant manual ingestion
import asyncio
import os
import argparse # <-- NEW: Essential for command-line MVP
import logging

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# Setup basic logging for production visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-script")

async def ingest_documents_async(site_id: str, data_dir: str, chroma_dir: str):
    """Processes a single tenant's data into their vector DB."""
    
    # [SCALABILITY CHECK]: Log which tenant we are processing
    logger.info(f"--- Starting Ingestion for Tenant: '{site_id}' ---")
    logger.info(f"Source Data: {data_dir}")
    logger.info(f"Output Chroma Dir: {chroma_dir}")

    all_documents = []

    # Source Validation - fine as is
    if not os.path.exists(data_dir):
        logger.error(f"Error: Data folder '{data_dir}' not found. Ingestion aborted.")
        return

    # PDF/TXT Loader - fine as is, async usage is correct
    logger.info(f"Scanning '{data_dir}' for .pdf and .txt files...")
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(filepath)
                documents = await loader.aload()
                all_documents.extend(documents)
                logger.info(f"Loaded PDF: {filepath} ({len(documents)} pages)")
            except Exception as exc:
                logger.error(f"Error loading PDF '{filepath}': {exc}")

        elif filename.lower().endswith(".txt"):
            try:
                loader = TextLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded TXT: {filepath}")
            except Exception as exc:
                logger.error(f"Error loading TXT '{filepath}': {exc}")

    if not all_documents:
        logger.warning(f"No documents found to ingest for tenant '{site_id}'.")
        return

    # Text Splitter - fine as is for MVP
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    logger.info(f"Split documents into {len(docs)} chunks.")

    # Output Validation & Persistence - fine as is
    try:
        os.makedirs(chroma_dir, exist_ok=True)
        # We are still using the OFFLINE Chroma builder here (creating SQLite files).
        # We'll migrate the 'tenants' directory manually for Phase 1 ship.
        await Chroma.afrom_documents(docs, embeddings, persist_directory=chroma_dir)
        logger.info(f"--- Successfully ingested data into '{chroma_dir}'. ---")
    except Exception as exc:
        logger.error(f"Error during ingestion for tenant '{site_id}': {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tenant MVP Data Ingest Script")
    
    # REQUIRED arguments for early MV-MVP operation
    parser.add_argument("--site_id", required=True, help="Site ID (tenant) matching sites.json")
    
    # OPTIONAL arguments, defaulting to common MVP patterns
    parser.add_argument("--data_dir", default="./data", help="Directory containing source data (PDF/TXT)")
    parser.add_argument("--chroma_dir", help="Output Chroma DB directory (persisted folder)")

    args = parser.parse_args()
    
    # -- MVP LOGIC --
    # 1. Resolve Data Directory. MVP pattern is often `./data/{site_id}`.
    # We default to just `./data` for Phase 1 ship, manually managing the split.
    resolved_data_dir = args.data_dir
    
    # 2. Resolve Chroma Directory. MVP pattern is `./tenants/{site_id}/chroma`.
    # We provide a default if not specified.
    resolved_chroma_dir = args.chroma_dir or f"./tenants/{args.site_id}/chroma"
    
    asyncio.run(ingest_documents_async(args.site_id, resolved_data_dir, resolved_chroma_dir))