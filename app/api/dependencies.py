# app/api/dependencies.py
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Dict

import httpx
from fastapi import FastAPI

from app.agents.s5_search_agent import S5SearchAgent
from app.config.base import settings
from app.connectors.base_connector import BaseConnector
# Import all your implemented connectors
from app.connectors.youtube_connector import YouTubeConnector
# from app.connectors.twitter_connector import TwitterConnector # <-- Add as you build them
from app.domains.search.schemas import Platform
from app.domains.search.service import SearchService

# This dictionary will hold our live application state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on application startup
    print("--- Starting Project Oracle Agent S5 ---")
    app_state["http_client"] = httpx.AsyncClient(timeout=30.0)
    # Initialize the agent on startup
    app_state["s5_agent"] = get_s5_agent()
    print("--- Agent Initialized. Service is ready. ---")
    
    yield
    
    # Runs on application shutdown
    print("--- Shutting down service ---")
    await app_state["s5_agent"].shutdown()
    await app_state["http_client"].aclose()
    print("--- Shutdown complete ---")


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    """Provides a singleton instance of the SearchService."""
    return SearchService()


@lru_cache(maxsize=1)
def get_connectors() -> Dict[Platform, BaseConnector]:
    """
    Initializes and returns a cached dictionary of all available connectors.
    """
    client = app_state["http_client"]
    
    connectors = {}
    try:
        # We wrap each connector in a try/except block so that a missing
        # API key for one service doesn't prevent the whole app from starting.
        connectors[Platform.YOUTUBE] = YouTubeConnector(settings=settings, client=client)
    except ValueError as e:
        print(f"Warning: Could not initialize YouTubeConnector: {e}")

    # Add other connectors here as they are built
    # try:
    #     connectors[Platform.TWITTER] = TwitterConnector(settings=settings, client=client)
    # except ValueError as e:
    #     print(f"Warning: Could not initialize TwitterConnector: {e}")
        
    return connectors


@lru_cache(maxsize=1)
def get_s5_agent() -> S5SearchAgent:
    """
    Provides a singleton instance of the S5SearchAgent for the application's lifespan.
    This ensures that its internal state (like cache and performance metrics) is maintained.
    """
    return S5SearchAgent(
        connectors=get_connectors(),
        search_service=get_search_service()
        # We can configure the agent from settings here if needed
        # cache_ttl_seconds=settings.CACHE_TTL,
    )