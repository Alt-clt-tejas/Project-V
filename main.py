# app/main.py
from fastapi import FastAPI

from app.api.dependencies import lifespan
from app.api.v1.routes import search_routes
from app.config.base import settings

# Create the main FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan  # Manages startup/shutdown events
)

# Include the API router for version 1
app.include_router(
    search_routes.router,
    prefix="/api/v1/search",
    tags=["Search"]
)


@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": f"Welcome to {settings.APP_NAME}"}