# app/api/dependencies.py
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Dict

import httpx
from fastapi import FastAPI

from app.agents.s5_search_agent import S5SearchAgent
from app.config.base import settings
from app.connectors.base_connector import BaseConnector
from app.connectors.youtube_connector import YouTubeConnector
from app.domains.search.schemas import Platform
from app.domains.search.service import SearchService


# This dictionary will hold our live application state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on application startup
    app_state["http_client"] = httpx.AsyncClient()
    yield
    # Runs on application shutdown
    await app_state["http_client"].aclose()


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return SearchService()


@lru_cache(maxsize=1)
def get_connectors() -> Dict[Platform, BaseConnector]:
    """
    Initializes and returns a dictionary of all available connectors.
    This is cached to avoid re-creating connectors on every request.
    """
    client = app_state["http_client"]
    
    # We can add more connectors here as they are implemented
    connectors = {
        Platform.YOUTUBE: YouTubeConnector(settings=settings, client=client),
        # Platform.TWITTER: TwitterConnector(settings=settings, client=client),
    }
    return connectors


def get_s5_agent() -> S5SearchAgent:
    """
    Dependency provider for the S5SearchAgent.
    FastAPI will call this function for every request that depends on it.
    """
    return S5SearchAgent(
        connectors=get_connectors(),
        search_service=get_search_service()
    )