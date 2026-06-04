import numpy as np
import logging
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# Global state for loaded references
_labels_array: Optional[np.ndarray] = None
_is_ready: bool = False

def load_references() -> bool:
    global _labels_array, _is_ready

    try:
        logger.info(f"Loading labels from {config.LABELS_ARRAY_PATH}")
        _labels_array = np.load(config.LABELS_ARRAY_PATH, mmap_mode="r")
        _is_ready = True
        logger.info("Labels loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load labels: {e}")
        _is_ready = False
        return False

def is_ready() -> bool:
    return _is_ready

def get_labels() -> Optional[np.ndarray]:
    return _labels_array

def cleanup():
    global _labels_array, _is_ready
    _labels_array = None
    _is_ready = False
    logger.info("References cleaned up")
