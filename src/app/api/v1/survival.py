from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple

from src.survival.cox_model import get_lifetime_pd

router = APIRouter()

class SurvivalRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        example={
            "RevolvingUtilizationOfUnsecuredLines": 0.35,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 1,
            "DebtRatio": 0.40,
            "MonthlyIncome": 6000.0,
            "NumberOfOpenCreditLinesAndLoans": 6,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2
        }
    )
    horizon_months: int = Field(36, ge=12, le=60)

class SurvivalResponse(BaseModel):
    concordance_index: float
    pd_12m: float
    pd_24m: float
    pd_36m: float
    ifrs9_stage: str
    survival_curve: List[Tuple[int, float]]

@router.post(
    "/lifetime-pd",
    response_model=SurvivalResponse,
    summary="IFRS 9 Lifetime PD & Staging (Cox Proportional Hazards)",
    description="Compute cumulative lifetime PD over 12, 24, and 36 months, IFRS 9 staging classification (Stage 1, 2, or 3), and monthly survival curve."
)
async def compute_lifetime_pd(payload: SurvivalRequest):
    try:
        res = get_lifetime_pd(payload.features, horizon_months=payload.horizon_months)
        return SurvivalResponse(
            concordance_index=res["c_index"],
            pd_12m=res["pd_12m"],
            pd_24m=res["pd_24m"],
            pd_36m=res["pd_36m"],
            ifrs9_stage=res["ifrs9_stage"],
            survival_curve=res["survival_curve"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
