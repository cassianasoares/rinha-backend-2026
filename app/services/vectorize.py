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

# Dicionário de risco MCC
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

def limitar(valor: float) -> float:
    """Limita valor entre 0 e 1."""
    return max(0.0, min(1.0, valor))

def vectorize_transaction(payload) -> np.ndarray:
    tx = payload.transaction
    cust = payload.customer
    merch = payload.merchant
    term = payload.terminal
    last_tx = payload.last_transaction

    logger.info(
        f"merch: {merch.id}, known_merchants: {cust.known_merchants}, "
        f"mcc: {merch.mcc}, mcc_risk: {mcc_risk.get(merch.mcc, 'unknown')}"
    )

    # 0 - amount
    amount = limitar(tx.amount / MAX_AMOUNT)

    # 1 - installments
    installments = limitar(tx.installments / MAX_INSTALLMENTS)

    # 2 - amount_vs_avg
    amount_vs_avg = limitar((tx.amount / cust.avg_amount) / AMOUNT_VS_AVG_RATIO) if cust.avg_amount > 0 else 0.0

    # 3 - hour_of_day
    hour_of_day = tx.requested_at.hour / 23.0

    # 4 - day_of_week (seg=0, dom=6)
    day_of_week = tx.requested_at.weekday() / 6.0

    # 5 - minutes_since_last_tx
    if last_tx:
        delta_minutes = (tx.requested_at - last_tx.timestamp).total_seconds() / 60.0
        minutes_since_last_tx = limitar(delta_minutes / MAX_MINUTES)
    else:
        minutes_since_last_tx = -1.0

    # 6 - km_from_last_tx
    km_from_last_tx = limitar(last_tx.km_from_current / MAX_KM) if last_tx else -1.0

    # 7 - km_from_home
    km_from_home = limitar(term.km_from_home / MAX_KM)

    # 8 - tx_count_24h
    tx_count_24h = limitar(cust.tx_count_24h / MAX_TX_COUNT_24H)

    # 9 - is_online
    is_online = 1.0 if term.is_online else 0.0

    # 10 - card_present
    card_present = 1.0 if term.card_present else 0.0

    # 11 - unknown_merchant
    unknown_merchant = 0.0 if merch.id in cust.known_merchants else 1.0

    # 12 - mcc_risk
    mcc_risk_value = mcc_risk.get(merch.mcc, 0.5)

    # 13 - merchant_avg_amount
    merchant_avg_amount = limitar(merch.avg_amount / MAX_MERCHANT_AVG_AMOUNT)

    return np.array([
        amount,
        installments,
        amount_vs_avg,
        hour_of_day,
        day_of_week,
        minutes_since_last_tx,
        km_from_last_tx,
        km_from_home,
        tx_count_24h,
        is_online,
        card_present,
        unknown_merchant,
        mcc_risk_value,
        merchant_avg_amount
    ], dtype=float)
