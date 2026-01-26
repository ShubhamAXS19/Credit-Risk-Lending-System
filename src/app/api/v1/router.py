from fastapi import APIRouter

api_router = APIRouter()


from src.app.api.v1 import predict

api_router.include_router(predict.router, tags=["prediction"])

