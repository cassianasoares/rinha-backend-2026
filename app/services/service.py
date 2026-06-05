from asyncio.log import logger
from config import config
import numpy as np

from services import search
from models.models import FraudScoreResponse, TransactionPayload
from services import references
from services.vectorize import vectorize_transaction

def score_transaction(payload: TransactionPayload) -> FraudScoreResponse:
    # Step 1: Vectorize the transaction
    query_vector = vectorize_transaction(payload)

    # Step 2: Search FAISS index
    index = search.get_global_index()
    neighbor_ids, distances = search.search_neighbors(index, query_vector, k=config.TOP_K_NEIGHBORS)

    # Step 3: Vectorized fraud count
    labels_array = references.get_labels()

    # Ensure neighbor_ids are valid and within bounds
    valid_mask = neighbor_ids >= 0
    valid_ids = neighbor_ids[valid_mask]

    if len(valid_ids) == 0:
        return FraudScoreResponse(approved=True, fraud_score=0.0)

    fraud_count = np.sum(labels_array[valid_ids] == config.FRAUD_LABEL_INT)

    # Step 4: Calculate fraud score and approval
    actual_neighbors = len(valid_ids)
    fraud_score_value = float(fraud_count) / actual_neighbors
    approved = fraud_score_value < config.FRAUD_THRESHOLD

    return FraudScoreResponse(
        approved=approved,
        fraud_score=fraud_score_value
    )
