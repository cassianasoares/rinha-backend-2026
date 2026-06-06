# Configuration constants for Fraud Score API
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Normalization thresholds (from data-model.md)
MAX_AMOUNT = 10000.0
MAX_INSTALLMENTS = 12.0
MAX_DISTANCE_KM = 1000.0
MAX_TX_COUNT_24H = 20.0
MAX_MINUTES_LAST_TX = 1440.0  # 24 hours
MAX_MERCHANT_AMOUNT = 10000.0

# FAISS index parameters
FAISS_DIMENSION = 14
FAISS_NLIST = 150
FAISS_NPROBE = 2
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index.bin"

# Fraud scoring thresholds
FRAUD_THRESHOLD = 0.6  # fraud_score >= 0.6 => rejected
FRAUD_LABEL_KEY = "fraud"
TOP_K_NEIGHBORS = 5
FRAUD_LABEL_INT = 1

# Model defaults
MCC_RISK_DEFAULT = 0.5  # Default MCC risk if unknown merchant

# Server configuration
READY_STATUS_KEY = "ready"
LABELS_ARRAY_PATH = BASE_DIR / "data" / "labels.npy"