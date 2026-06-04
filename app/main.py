"""FastAPI application entry point.

Manages the full lifecycle: reference data loading, FAISS index
construction, and cleanup on shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services import references
from services import search
from config import config
from controller import api

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup loads references and builds FAISS index."""
    logger.info("Application startup")

    # Phase 1: Load reference data (memmap)
    success = references.load_references()
    if not success:
        logger.error("Failed to load references — service will start in degraded mode")

    # Phase 2: Load FAISS index
    if success:
        index = search.load_index(config.FAISS_INDEX_PATH)
        if index is None:
            logger.error(f"Persisted FAISS index not found at {config.FAISS_INDEX_PATH}")
        else:
            search.set_global_index(index)
            logger.info(f"FAISS index ready with {index.ntotal} vectors")

    yield

    logger.info("Application shutdown")
    search.cleanup()
    references.cleanup()


app = FastAPI(
    title="Fraud Score API",
    description="API for fraud risk scoring using FAISS nearest neighbor search",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api.router)


@app.get("/", tags=["root"])
async def root():
    return {"message": "Fraud Score API v1.0"}
