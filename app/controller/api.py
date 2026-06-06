"""API routes for the Fraud Score API.

Endpoints:
- GET  /ready       — Health check / readiness probe
- POST /fraud-score — Transaction fraud scoring
"""

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from services import references
from models.models import FraudScoreResponse, TransactionPayload
from services.service import score_transaction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/ready", tags=["health"])
async def ready():
    if references.is_ready():
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="References not initialized")


@router.post("/fraud-score", response_model=FraudScoreResponse, tags=["scoring"])
async def fraud_score(payload: TransactionPayload):
    # Marca o início da execução
    start = time.perf_counter()

    # Executa a lógica principal (FAISS, etc.) em threadpool
    result = await run_in_threadpool(score_transaction, payload)

    # Marca o fim e calcula tempo
    elapsed = time.perf_counter() - start
    logger.info(f"/fraud-score execution time: {elapsed*1000:.2f} ms")

    return result
