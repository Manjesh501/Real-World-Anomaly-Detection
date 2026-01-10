from pydantic import BaseModel, Field, constr, conint, confloat
from typing import List, Literal, Optional


class TransactionRequest(BaseModel):
    """Incoming raw transaction as seen by the bank in real time."""

    customer: constr(min_length=1) = Field(..., description="Unique customer identifier")
    age: constr(min_length=1) = Field(..., description="Age bucket as string, e.g. '1','2',... as in BankSim")
    gender: constr(min_length=1) = Field(..., description="Customer gender, e.g. 'M' or 'F'")
    zipcodeOri: constr(min_length=1) = Field(..., description="Customer origin zipcode")
    merchant: constr(min_length=1) = Field(..., description="Merchant identifier")
    zipMerchant: constr(min_length=1) = Field(..., description="Merchant zipcode")
    category: constr(min_length=1) = Field(..., description="Merchant category, e.g. 'es_transportation'")
    amount: confloat(ge=0) = Field(..., description="Transaction amount in euros")
    step: conint(ge=0) = Field(..., description="Time step (hours since simulation start)")


class PredictionResponse(BaseModel):
    """Model prediction and explanation for a single transaction."""

    is_fraud: bool = Field(..., description="True if transaction is predicted as fraud")
    fraud_score: float = Field(
        ..., description="Anomaly score derived from Isolation Forest (higher = more anomalous)"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Qualitative risk bucket derived from fraud_score and model decision"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons contributing to the risk assessment",
    )
    data_quality_flag: Optional[str] = Field(
        default=None,
        description="Warning if customer/merchant not in training data (None = known, 'unseen_customer' = new customer, 'unseen_merchant' = new merchant)"
    )
