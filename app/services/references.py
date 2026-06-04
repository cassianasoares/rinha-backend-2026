import gzip
import json
import numpy as np
import logging
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# Global state for loaded references
_vectors_memmap: Optional[np.ndarray] = None
_labels_array: Optional[np.ndarray] = None
_ids_array: Optional[np.ndarray] = None
_is_ready: bool = False


def load_references(gz_path: str = config.REFERENCE_DATA_PATH) -> bool:
    global _vectors_memmap, _labels_array, _ids_array, _is_ready

    try:
        logger.info(f"Loading references from {gz_path}")

        vectors = []
        labels = []
        ids = []

        # O arquivo é um JSON único (lista), não linha por linha
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        for idx, record in enumerate(data):
            vectors.append(record["vector"])
            labels.append(record["label"])
            ids.append(idx)  # gera id sequencial

        logger.info(f"Loaded {len(vectors)} reference records")

        vectors_array = np.array(vectors, dtype=np.float32)
        _labels_array = np.array(labels, dtype=object)
        _ids_array = np.array(ids, dtype=np.int32)

        # Persistência em disco
        np.save(config.VECTORS_MEMMAP_PATH, vectors_array)
        np.save(config.LABELS_ARRAY_PATH, _labels_array)
        np.save(config.IDS_ARRAY_PATH, _ids_array)

        # Memmap para vetores
        _vectors_memmap = np.load(config.VECTORS_MEMMAP_PATH, mmap_mode="r")

        _is_ready = True
        logger.info("References loaded and memmap initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        _is_ready = False
        return False


def is_ready() -> bool:
    return _is_ready


def get_vectors() -> Optional[np.ndarray]:
    return _vectors_memmap


def get_labels() -> Optional[np.ndarray]:
    return _labels_array


def get_ids() -> Optional[np.ndarray]:
    return _ids_array


def cleanup():
    global _vectors_memmap, _labels_array, _ids_array, _is_ready
    _vectors_memmap = None
    _labels_array = None
    _ids_array = None
    _is_ready = False
    logger.info("References cleaned up")
