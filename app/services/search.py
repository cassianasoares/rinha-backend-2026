import logging
from typing import Tuple
import os

import faiss
import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Global state for the FAISS index
_faiss_index: faiss.Index | None = None


def search_neighbors(
    index: faiss.Index,
    query_vector: np.ndarray,
    k: int = config.TOP_K_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray]:

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    # Clamp k ao número de vetores disponíveis
    k = min(k, index.ntotal)

    distances, ids = index.search(query_vector, k)

    return ids.flatten(), distances.flatten()


def save_index(index: faiss.Index, path: str = config.FAISS_INDEX_PATH) -> None:
    faiss.write_index(index, path)
    logger.info(f"FAISS index saved to {path}")


def load_index(path):
    if not os.path.exists(path):
        return None
    try:
        index = faiss.read_index(str(path))
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None



def set_global_index(index: faiss.Index) -> None:
    global _faiss_index
    _faiss_index = index


def get_global_index() -> faiss.Index | None:
    return _faiss_index


def cleanup() -> None:
    global _faiss_index
    _faiss_index = None
    logger.info("FAISS index cleaned up")
