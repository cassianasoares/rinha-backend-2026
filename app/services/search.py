"""FAISS-based vector search module for fraud scoring.

Builds a FAISS index from reference vectors and supports top-k neighbor
search using L2 (Euclidean) distance metric.
"""

import logging
from typing import Tuple
from pathlib import Path

import faiss
import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Global state for the FAISS index
_faiss_index: faiss.Index | None = None


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    n_vectors, dimension = vectors.shape
    assert dimension == config.FAISS_DIMENSION, (
        f"Expected {config.FAISS_DIMENSION}-d vectors, got {dimension}-d"
    )

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if n_vectors >= config.FAISS_NLIST:
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, config.FAISS_NLIST, faiss.METRIC_L2)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = config.FAISS_NPROBE
        logger.info(
            f"Built IndexIVFFlat with {n_vectors} vectors, "
            f"nlist={config.FAISS_NLIST}, nprobe={config.FAISS_NPROBE}"
        )
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        logger.info(f"Built IndexFlatL2 with {n_vectors} vectors (dataset too small for IVF)")

    return index


def search_neighbors(
    index: faiss.Index,
    query_vector: np.ndarray,
    k: int = config.TOP_K_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index for the k nearest neighbors."""

    # Verificação extra para garantir que index é válido
    if not isinstance(index, faiss.Index):
        raise TypeError(f"Expected faiss.Index, got {type(index)}")

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
    index = faiss.read_index(path)
    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = config.FAISS_NPROBE
    logger.info(f"FAISS index loaded from {path} ({index.ntotal} vectors)")
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
