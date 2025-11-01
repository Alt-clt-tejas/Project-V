# app/api/dependencies.py
import logging
from functools import lru_cache
from typing import Dict
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends

from app.config.base import AppSettings
from app.connectors.base_connector import BaseConnector
from app.connectors.youtube_connector import YouTubeConnector
from app.connectors.instagram_connector import InstagramConnector
from app.domains.search.schemas import Platform
from app.domains.search.service import SearchService
from app.agents.s5_search_agent import S5SearchAgent

logger = logging.getLogger(__name__)
app_state = {}

@asynccontextmanager
async def lifespan(app):
    """Application startup/shutdown lifecycle manager."""
    try:
        logger.info("Starting application...")
        app_state["http_client"] = httpx.AsyncClient(timeout=30.0)
        app_state["s5_agent"] = get_s5_agent()  # Initialize the agent singleton on startup
        logger.info("Application startup complete.")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down application...")
        try:
            # Clean up agent
            agent = app_state.get("s5_agent")
            if agent and hasattr(agent, "shutdown"):
                await agent.shutdown()

            # Clean up HTTP client
            client = app_state.get("http_client")
            if client:
                await client.aclose()

            # Clear app state
            app_state.clear()
            
            logger.info("Application shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            # Don't raise during shutdown

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Lazily load and cache application settings."""
    return AppSettings()

def get_youtube_connector() -> YouTubeConnector:
    """Get YouTube connector instance - called per request."""
    settings = get_settings()
    client = app_state.get("http_client")
    if not client:
        raise RuntimeError("HTTP client not initialized. Make sure the app has started properly.")
    return YouTubeConnector(settings=settings, client=client)

@lru_cache(maxsize=1)
def get_connectors() -> Dict[Platform, BaseConnector]:
    """Initialize and cache platform connectors."""
    settings = get_settings()
    http_client = app_state.get("http_client")
    if not http_client:
        raise RuntimeError("HTTP client is not initialized.")

    connectors: Dict[Platform, BaseConnector] = {}
    if settings.YOUTUBE_API_KEY:
        connectors[Platform.YOUTUBE] = YouTubeConnector(settings=settings, client=http_client)
    if settings.INSTAGRAM_USERNAME and settings.INSTAGRAM_PASSWORD:
        connectors[Platform.INSTAGRAM] = InstagramConnector(settings=settings, client=http_client)
    
    logger.info(f"Initialized connectors: {[p.value for p in connectors.keys()]}")
    return connectors

@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    """Provide a singleton instance of the SearchService."""
    return SearchService()

@lru_cache(maxsize=1)
def get_s5_agent() -> S5SearchAgent:
    """Create and provide a singleton instance of the S5SearchAgent."""
    return S5SearchAgent(
        connectors=get_connectors(),
        search_service=get_search_service()
    )

def get_agent() -> S5SearchAgent:
    """Dependency provider that retrieves the agent instance from the app state."""
    agent = app_state.get("s5_agent")
    if not agent:
        raise RuntimeError("S5SearchAgent not found in application state.")
    return agent