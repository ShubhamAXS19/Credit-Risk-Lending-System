from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.api.v1.router import api_router

app = FastAPI(title="Credit Risk & Lending System API")

# CORS middleware to allow cross-origin requests (useful if frontend runs on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "app_name": "Credit Risk & Lending System"}
