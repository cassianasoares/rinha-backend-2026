from asyncio.log import logger
from config import config
import numpy as np
import time

from services import search
from models.models import FraudScoreResponse, TransactionPayload
from services import references
from services.vectorize import vectorize_transaction

def score_transaction(payload: TransactionPayload) -> FraudScoreResponse:
    t0 = time.perf_counter()
    query_vector = vectorize_transaction(payload)

    logger.warning(f"Query vector: {query_vector}. Id: {payload.id}")

    t_vectorize = time.perf_counter() - t0

    index = search.get_global_index()
    t1 = time.perf_counter()
    neighbor_ids, distances = search.search_neighbors(index, query_vector, k=config.TOP_K_NEIGHBORS)
    t_faiss = time.perf_counter() - t1

    labels_array = references.get_labels()
    valid_mask = neighbor_ids >= 0
    valid_ids = neighbor_ids[valid_mask]

    if len(valid_ids) == 0:
        return FraudScoreResponse(approved=True, fraud_score=0.0)

    fraud_count = np.sum(labels_array[valid_ids] == config.FRAUD_LABEL_INT)
    actual_neighbors = len(valid_ids)
    fraud_score_value = float(fraud_count) / actual_neighbors
    approved = fraud_score_value < config.FRAUD_THRESHOLD

    logger.info(f"Vectorize: {t_vectorize*1000:.2f} ms | FAISS: {t_faiss*1000:.2f} ms")

    return FraudScoreResponse(
        approved=approved,
        fraud_score=fraud_score_value
    )

