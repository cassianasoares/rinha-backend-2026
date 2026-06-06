import gzip
import json
import numpy as np
import faiss
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configs
FAISS_NLIST = 150        # número de listas (clusters)
FAISS_NPROBE = 2         # número de listas exploradas na busca
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "references.json.gz")
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
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    labels_int = [1 if label == "fraud" else 0 for label in labels]
    labels_array = np.array(labels_int, dtype=np.int8)
    np.save(LABELS_ARRAY_PATH, labels_array)
    logger.info(f"Saved labels to {LABELS_ARRAY_PATH}")
    
    vectors_array = np.array(vectors, dtype=np.float32)
    n_vectors, dimension = vectors_array.shape
    
    logger.info(f"Building FAISS IndexIVFScalarQuantizer (QT_8bit) for {n_vectors} vectors, dim={dimension}...")
    quantizer = faiss.IndexFlatL2(dimension)
    
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dimension,
        FAISS_NLIST,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_L2
    )
    
    index.train(vectors_array)
    index.add(vectors_array)
    index.nprobe = FAISS_NPROBE
    
    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("Pre-processing complete.")

if __name__ == "__main__":
    main()
