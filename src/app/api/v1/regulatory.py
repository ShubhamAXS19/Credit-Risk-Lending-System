from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.basel.pd_model import platt_scale, map_pd_to_grade
from src.basel.lgd_model import get_lgd, CollateralType
from src.basel.ead_model import calculate_ead, LoanType
from src.basel.expected_loss import calculate_expected_loss, calculate_rwa

router = APIRouter()

class RegulatoryAssessmentRequest(BaseModel):
    applicant_id: Optional[str] = "LOAN-10001"
    raw_default_prob: float = Field(0.08, ge=0.0, le=1.0)
    drawn_amount: float = Field(15000.0, ge=0.0)
    undrawn_amount: float = Field(5000.0, ge=0.0)
    loan_type: str = Field("TERM_LOAN", example="TERM_LOAN")
    collateral_type: str = Field("UNSECURED", example="UNSECURED")
    ltv_ratio: float = Field(1.0, ge=0.0)
    horizon_months: int = Field(12, ge=1)
    use_ml_lgd: bool = False

class RegulatoryAssessmentResponse(BaseModel):
    applicant_id: str
    calibrated_pd: float
    basel_risk_grade: str
    lgd: float
    ead: float
    expected_loss: float
    rwa: float
    tier1_capital_required_8pct: float
    capital_buffer_required_10_5pct: float

@router.post(
    "/regulatory",
    response_model=RegulatoryAssessmentResponse,
    summary="Basel II/III Regulatory Credit Risk Assessment",
    description="Compute Basel II/III risk metrics (Calibrated PD, LGD, EAD, Expected Loss, Risk-Weighted Assets, and Capital Requirements) per loan application."
)
async def compute_regulatory_assessment(payload: RegulatoryAssessmentRequest):
    try:
        calibrated_pd = platt_scale(payload.raw_default_prob)
        risk_grade = map_pd_to_grade(calibrated_pd).value
        
        lgd_val = get_lgd(
            loan_id=payload.applicant_id,
            collateral_type=payload.collateral_type,
            ltv_ratio=payload.ltv_ratio,
            use_ml=payload.use_ml_lgd
        )
        
        ead_val = calculate_ead(
            outstanding_balance=payload.drawn_amount,
            undrawn_commitment=payload.undrawn_amount,
            loan_type=payload.loan_type,
            horizon_months=payload.horizon_months
        )
        
        el_val = calculate_expected_loss(calibrated_pd, lgd_val, ead_val)
        rwa_val = calculate_rwa(calibrated_pd, lgd_val, ead_val)
        
        cap_8pct = round(rwa_val * 0.08, 2)
        cap_10_5pct = round(rwa_val * 0.105, 2)
        
        return RegulatoryAssessmentResponse(
            applicant_id=payload.applicant_id,
            calibrated_pd=round(calibrated_pd, 4),
            basel_risk_grade=risk_grade,
            lgd=round(lgd_val, 4),
            ead=round(ead_val, 2),
            expected_loss=round(el_val, 2),
            rwa=round(rwa_val, 2),
            tier1_capital_required_8pct=cap_8pct,
            capital_buffer_required_10_5pct=cap_10_5pct
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
