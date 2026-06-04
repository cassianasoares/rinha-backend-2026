from asyncio.log import logger
from config import config
import time
from typing import List

from services import search
from models.models import FraudScoreResponse, NeighborInfo, TransactionPayload
from services import references
from services.vectorize import vectorize_transaction


def score_transaction(payload: TransactionPayload, start_time: float) -> FraudScoreResponse:
    # Step 1: Vectorize the transaction
    query_vector = vectorize_transaction(payload)

    # Step 2: Buscar o índice FAISS global corretamente
    index = search.get_global_index()   # ✅ com parênteses, retorna o objeto FAISS
    neighbor_ids, distances = search.search_neighbors(index, query_vector, k=config.TOP_K_NEIGHBORS)

    # Step 3: Look up labels and IDs for the neighbors
    labels_array = references.get_labels()
    ids_array = references.get_ids()

    neighbors: List[NeighborInfo] = []
    fraud_count = 0

    for idx, dist in zip(neighbor_ids, distances):
        idx = int(idx)
        if idx < 0:
            continue
        neighbor_label = str(labels_array[idx])
        neighbor_id = str(ids_array[idx])

        if neighbor_label == config.FRAUD_LABEL_KEY:
            fraud_count += 1

        neighbors.append(
            NeighborInfo(
                neighbor_id=neighbor_id,
                label=neighbor_label,
                distance=float(dist),
            )
        )

    # Step 4: Calculate fraud score and approval
    actual_neighbors = len(neighbors)
    fraud_score_value = fraud_count / actual_neighbors if actual_neighbors > 0 else 0.0
    approved = fraud_score_value < config.FRAUD_THRESHOLD

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Scored transaction {payload.id}: "
        f"fraud_score={fraud_score_value:.2f}, approved={approved}, "
        f"fraud_neighbors={fraud_count}/{actual_neighbors}, "
        f"elapsed={elapsed_ms:.1f}ms"
    )

    return FraudScoreResponse(
        approved=approved,
        fraud_score=fraud_score_value,
        neighbors=neighbors,
    )
