"""API routes for the Fraud Score API.

Endpoints:
- GET  /ready       — Health check / readiness probe
- POST /fraud-score — Transaction fraud scoring
"""

import logging
import time

from fastapi import APIRouter, HTTPException

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
def fraud_score(payload: TransactionPayload):
    return score_transaction(payload)
