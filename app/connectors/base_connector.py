# app/connectors/base_connector.py
import abc
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.base import AppSettings
from app.domains.search.schemas import CreatorProfile, Platform


class BaseConnector(abc.ABC):
    """
    Abstract Base Class for all platform connectors.
    It defines a common interface for searching and provides a shared,
    resilient HTTP client.
    """
    def __init__(self, settings: AppSettings, client: httpx.AsyncClient):
        """
        Initializes the connector with settings and an async HTTP client.

        Args:
            settings: The application settings instance.
            client: An instance of httpx.AsyncClient for making API calls.
        """
        self.settings = settings
        self.client = client

    @property
    @abc.abstractmethod
    def platform(self) -> Platform:
        """The platform identifier for this connector."""
        raise NotImplementedError

    @abc.abstractmethod
    async def search(self, query: str) -> List[CreatorProfile]:
        """
        Performs a search for a creator on the specific platform.

        Args:
            query: The search query (name, handle, etc.).

        Returns:
            A list of standardized CreatorProfile objects.
        """
        raise NotImplementedError

    # You can add shared utility methods here. For example, a resilient request method.
    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(3))
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        A shared, resilient method for making HTTP requests with retries.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            url: The URL to request.
            **kwargs: Additional arguments for httpx.AsyncClient.request.

        Returns:
            An httpx.Response object.

        Raises:
            httpx.HTTPStatusError: If the response status code indicates an error.
        """
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()  # Raises an exception for 4xx/5xx responses
        return response