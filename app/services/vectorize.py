from asyncio.log import logger
import numpy as np
from typing import Dict

# Constantes
MAX_AMOUNT = 10000
MAX_INSTALLMENTS = 12
AMOUNT_VS_AVG_RATIO = 10
MAX_MINUTES = 1440
MAX_KM = 1000
MAX_TX_COUNT_24H = 20
MAX_MERCHANT_AVG_AMOUNT = 10000

# Risk Dictionary MCC
mcc_risk: Dict[str, float] = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}

def vectorize_transaction(payload) -> np.ndarray:
    # Extract nested objects to minimize Pydantic access overhead
    tx = payload.transaction
    cust = payload.customer
    merch = payload.merchant
    term = payload.terminal
    last_tx = payload.last_transaction

    # Pre-fetch values to local variables
    tx_amount = tx.amount
    tx_inst = tx.installments
    tx_time = tx.requested_at
    cust_avg = cust.avg_amount
    cust_24h = cust.tx_count_24h
    merch_id = merch.id
    merch_mcc = merch.mcc
    merch_avg = merch.avg_amount
    term_home_km = term.km_from_home
    term_online = term.is_online
    term_present = term.card_present

    # 0 - amount
    amount = max(0.0, min(1.0, tx_amount / MAX_AMOUNT))
    # 1 - installments
    installments = max(0.0, min(1.0, tx_inst / MAX_INSTALLMENTS))
    # 2 - amount_vs_avg
    amount_vs_avg = max(0.0, min(1.0, (tx_amount / cust_avg) / AMOUNT_VS_AVG_RATIO)) if cust_avg > 0 else 0.0
    # 3 - hour_of_day
    hour_of_day = tx_time.hour / 23.0
    # 4 - day_of_week
    day_of_week = tx_time.weekday() / 6.0

    if last_tx:
        delta_minutes = (tx_time - last_tx.timestamp).total_seconds() / 60.0
        minutes_since_last_tx = max(0.0, min(1.0, delta_minutes / MAX_MINUTES))
        km_from_last_tx = max(0.0, min(1.0, last_tx.km_from_current / MAX_KM))
    else:
        minutes_since_last_tx = -1.0
        km_from_last_tx = -1.0

    km_from_home = max(0.0, min(1.0, term_home_km / MAX_KM))
    tx_count_24h = max(0.0, min(1.0, cust_24h / MAX_TX_COUNT_24H))
    is_online = 1.0 if term_online else 0.0
    card_present = 1.0 if term_present else 0.0
    unknown_merchant = 0.0 if merch_id in cust.known_merchants else 1.0
    mcc_risk_value = mcc_risk.get(merch_mcc, 0.5)
    merchant_avg_amount = max(0.0, min(1.0, merch_avg / MAX_MERCHANT_AVG_AMOUNT))

    return np.array([
        amount, installments, amount_vs_avg, hour_of_day, day_of_week,
        minutes_since_last_tx, km_from_last_tx, km_from_home, tx_count_24h,
        is_online, card_present, unknown_merchant, mcc_risk_value, merchant_avg_amount
    ], dtype=np.float32)
