# app/connectors/youtube_connector.py
import asyncio
from typing import List, Dict, Any

import httpx
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config.base import AppSettings
from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import CreatorProfile, Platform


class YouTubeConnector(BaseConnector):
    """Connector for the YouTube Data API v3."""

    def __init__(self, settings: AppSettings, client: httpx.AsyncClient):
        super().__init__(settings, client)
        if not self.settings.YOUTUBE_API_KEY:
            raise ValueError("YOUTUBE_API_KEY is not set in the configuration.")
        
        # The google-api-client is synchronous, so we build it once.
        # It will be used within asyncio.to_thread to avoid blocking.
        self.youtube_service = build(
            'youtube', 'v3', developerKey=self.settings.YOUTUBE_API_KEY.get_secret_value()
        )

    @property
    def platform(self) -> Platform:
        return Platform.YOUTUBE

    async def search(self, query: str) -> List[CreatorProfile]:
        """
        Searches for YouTube channels matching the query.
        
        Note: The YouTube Search API doesn't return subscriber counts directly.
        A follow-up call to the Channels.list endpoint would be needed per result,
        which can be costly. For the VET stage, this would be appropriate,
        but for initial SEARCH, we omit it to keep latency low.
        """
        try:
            # Run the synchronous SDK call in a separate thread
            search_response = await asyncio.to_thread(
                self.youtube_service.search().list(
                    q=query,
                    part='snippet',
                    type='channel',
                    maxResults=10  # Limit results to a reasonable number
                ).execute
            )
            return self._parse_response(search_response)
        except HttpError as e:
            # TODO: Add structured logging here
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            return []

    def _parse_response(self, response: Dict[str, Any]) -> List[CreatorProfile]:
        """Parses the raw API response into a list of CreatorProfile objects."""
        profiles = []
        for item in response.get('items', []):
            snippet = item.get('snippet', {})
            channel_id = snippet.get('channelId')

            if not channel_id:
                continue

            profiles.append(
                CreatorProfile(
                    platform=self.platform,
                    name=snippet.get('title', ''),
                    handle=snippet.get('channelTitle', ''), # Often same as title for channels
                    profile_url=f"https://www.youtube.com/channel/{channel_id}",
                    bio=snippet.get('description'),
                    # Subscriber count is not available in the search result.
                    # This would be fetched in a deeper analysis stage.
                    followers_count=None, 
                    is_verified=False # Verification status is not in search results
                )
            )
        return profiles