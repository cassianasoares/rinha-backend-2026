import logging
from typing import Tuple
from pathlib import Path

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


def load_index(path: str = config.FAISS_INDEX_PATH) -> faiss.Index | None:
    if not Path(path).exists():
        return None
    index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)

    # Ajusta nprobe se o índice suportar (IVF)
    if hasattr(index, "nprobe"):
        index.nprobe = config.FAISS_NPROBE
        logger.info(f"FAISS index loaded with nprobe={index.nprobe}")
    else:
        logger.info(f"FAISS index loaded (type={type(index).__name__})")

    return index


def set_global_index(index: faiss.Index) -> None:
    global _faiss_index
    _faiss_index = index


def get_global_index() -> faiss.Index | None:
    return _faiss_index


def cleanup() -> None:
    global _faiss_index
    _faiss_index = None
    logger.info("FAISS index cleaned up")
