# app/api/metrics.py
from fastapi import APIRouter
from app.utils.metrics import metrics_collector

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("/", response_model=dict)
async def get_metrics():
    return await metrics_collector.get_metrics()