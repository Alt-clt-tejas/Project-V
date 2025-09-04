# app/api/health.py - Health check endpoints
from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_connectors, get_settings
from app.config.base import AppSettings
from app.domains.search.schemas import Platform

router = APIRouter(prefix="/health", tags=["health"])

class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: float
    connectors: Dict[str, Any]
    
class ConnectorHealth(BaseModel):
    """Individual connector health status."""
    platform: str
    status: str
    last_check: datetime
    details: Dict[str, Any]

class SystemHealth(BaseModel):
    """Overall system health."""
    overall_status: str
    healthy_connectors: int
    total_connectors: int
    issues: List[str]

@router.get("/", response_model=HealthStatus)
async def get_health_status(
    settings: AppSettings = Depends(get_settings),
    connectors = Depends(get_connectors)
) -> HealthStatus:
    """
    Get overall system health status.
    Returns detailed information about all connectors and system status.
    """
    start_time = getattr(get_health_status, '_start_time', datetime.utcnow())
    if not hasattr(get_health_status, '_start_time'):
        get_health_status._start_time = start_time
    
    uptime = (datetime.utcnow() - start_time).total_seconds()
    
    connector_statuses = {}
    overall_healthy = True
    
    for platform, connector in connectors.items():
        try:
            if hasattr(connector, 'get_health_status'):
                status = await connector.get_health_status()
                connector_statuses[platform.value] = {
                    **status,
                    "last_check": datetime.utcnow().isoformat()
                }
            else:
                connector_statuses[platform.value] = {
                    "status": "unknown",
                    "message": "Health check not implemented",
                    "last_check": datetime.utcnow().isoformat()
                }
        except Exception as e:
            connector_statuses[platform.value] = {
                "status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
            overall_healthy = False
    
    return HealthStatus(
        status="healthy" if overall_healthy else "degraded",
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        connectors=connector_statuses
    )

@router.get("/connectors", response_model=List[ConnectorHealth])
async def get_connector_health(connectors = Depends(get_connectors)) -> List[ConnectorHealth]:
    """Get detailed health information for all connectors."""
    health_checks = []
    
    for platform, connector in connectors.items():
        try:
            if hasattr(connector, 'get_health_status'):
                details = await connector.get_health_status()
                status = details.get('status', 'unknown')
            else:
                details = {"message": "Health check not implemented"}
                status = "unknown"
                
            health_checks.append(ConnectorHealth(
                platform=platform.value,
                status=status,
                last_check=datetime.utcnow(),
                details=details
            ))
        except Exception as e:
            health_checks.append(ConnectorHealth(
                platform=platform.value,
                status="error",
                last_check=datetime.utcnow(),
                details={"error": str(e)}
            ))
    
    return health_checks

@router.get("/ready")
async def readiness_check(connectors = Depends(get_connectors)) -> Dict[str, Any]:
    """
    Kubernetes-style readiness check.
    Returns 200 if at least one connector is healthy, 503 otherwise.
    """
    healthy_count = 0
    total_count = len(connectors)
    
    for platform, connector in connectors.items():
        try:
            if hasattr(connector, 'get_health_status'):
                status = await connector.get_health_status()
                if status.get('status') == 'healthy':
                    healthy_count += 1
            else:
                healthy_count += 1  # Assume healthy if no health check
        except Exception:
            pass
    
    is_ready = healthy_count > 0
    
    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"No healthy connectors available ({healthy_count}/{total_count})"
        )
    
    return {
        "ready": True,
        "healthy_connectors": healthy_count,
        "total_connectors": total_count,
        "message": f"{healthy_count}/{total_count} connectors healthy"
    }

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes-style liveness check.
    Returns 200 if the application is running, regardless of connector status.
    """
    return {
        "alive": "true",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Application is running"
    }