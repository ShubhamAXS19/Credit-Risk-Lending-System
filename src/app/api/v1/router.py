from fastapi import APIRouter
from src.app.api.v1 import predict, explain, regulatory, stress, survival

api_router = APIRouter()

api_router.include_router(predict.router, tags=["prediction"])
api_router.include_router(explain.router, tags=["explainability"])
api_router.include_router(regulatory.router, tags=["regulatory-basel"])
api_router.include_router(stress.router, tags=["macro-stress-testing"])
api_router.include_router(survival.router, tags=["ifrs9-survival-analysis"])
