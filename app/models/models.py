from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class NeighborInfo(BaseModel):
    neighbor_id: str
    label: str
    distance: float


class FraudScoreResponse(BaseModel):
    approved: bool
    fraud_score: float = Field(..., ge=0.0, le=1.0)
#    neighbors: List[NeighborInfo] = []


class Transaction(BaseModel):
    amount: float
    installments: int
    requested_at: datetime


class Customer(BaseModel):
    avg_amount: float
    tx_count_24h: int
    known_merchants: List[str]


class Merchant(BaseModel):
    id: str
    mcc: str
    avg_amount: float


class Terminal(BaseModel):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransaction(BaseModel):
    timestamp: datetime
    km_from_current: float


class TransactionPayload(BaseModel):
    id: str
    transaction: Transaction
    customer: Customer
    merchant: Merchant
    terminal: Terminal
    last_transaction: Optional[LastTransaction] = None