import gzip
import json
import numpy as np
import faiss
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configs
FAISS_DIMENSION = 14
FAISS_NLIST = 100
FAISS_NPROBE = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DATA_PATH = os.path.join(DATA_DIR, "references.json.gz")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
LABELS_ARRAY_PATH = os.path.join(DATA_DIR, "labels.npy")

def main():
    logger.info(f"Loading references from {REFERENCE_DATA_PATH}")
    
    if not os.path.exists(REFERENCE_DATA_PATH):
        logger.error(f"File not found: {REFERENCE_DATA_PATH}")
        sys.exit(1)
        
    vectors = []
    labels = []
    
    with gzip.open(REFERENCE_DATA_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    for record in data:
        vectors.append(record["vector"])
        labels.append(record["label"])
        
    logger.info(f"Loaded {len(vectors)} records.")
    
    # Save labels as fixed-size string array to save memory
    labels_array = np.array(labels, dtype="S10")
    np.save(LABELS_ARRAY_PATH, labels_array)
    logger.info(f"Saved labels to {LABELS_ARRAY_PATH}")
    
    # Build FAISS index with quantization (int8)
    vectors_array = np.array(vectors, dtype=np.float32)
    n_vectors, dimension = vectors_array.shape
    
    logger.info(f"Building FAISS IndexIVFScalarQuantizer (QT_8bit) for {n_vectors} vectors...")
    quantizer = faiss.IndexFlatL2(dimension)
    
    # QT_8bit or QT_fp16
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dimension, FAISS_NLIST, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
    )
    
    index.train(vectors_array)
    index.add(vectors_array)
    index.nprobe = FAISS_NPROBE
    
    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("Pre-processing complete.")

if __name__ == "__main__":
    main()
