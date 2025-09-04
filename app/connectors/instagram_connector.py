# app/connectors/instagram_connector.py
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import httpx
from instagrapi import Client
from instagrapi.exceptions import (
    UserNotFound, PrivateError, LoginRequired, RateLimitError,
    PleaseWaitFewMinutes, ChallengeRequired, TwoFactorRequired
)

# --- FIXED IMPORT ---
# Import the AppSettings CLASS, not the non-existent 'settings' instance
from app.config.base import AppSettings
# --- END FIX ---

from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import (
    CreatorProfile, Platform, SocialMetrics, EngagementMetrics,
    ProfileMetadata, VerificationStatus
)
from app.exceptions import (
    AuthenticationException, RateLimitException, ConnectorException
)

logger = logging.getLogger(__name__)

class InstagramConnector(BaseConnector):
    """
    Unified Instagram connector with:
    - Robust session management & persistence
    - Rate limiting with adaptive backoff
    - Centralized exception handling
    - Support for private profiles (limited data)
    - Detailed health reporting
    """
    def __init__(self, settings: AppSettings, client: httpx.AsyncClient):
        super().__init__(settings, client)

        if not (settings.INSTAGRAM_USERNAME and settings.INSTAGRAM_PASSWORD):
            raise ValueError("INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD must be configured.")

        self._client = Client()
        self._username = settings.INSTAGRAM_USERNAME.get_secret_value()
        self._password = settings.INSTAGRAM_PASSWORD.get_secret_value()
        self._session_file = Path(settings.INSTAGRAM_SESSION_PATH) / f"{self._username}_session.json"
        
        # Session/login state
        self._logged_in = False
        self._session_lock = asyncio.Lock()
        self._login_attempts = 0
        self._last_login_attempt_time = 0

        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        self._min_delay = 1.0  # minimum delay between requests in seconds

    @property
    def platform(self) -> Platform:
        return Platform.INSTAGRAM

    async def _ensure_login(self) -> None:
        async with self._session_lock:
            if self._logged_in:
                return
            if await self._load_session():
                return
            await self._perform_login()
    
    async def _load_session(self) -> bool:
        if not self._session_file.exists():
            return False
        try:
            await asyncio.to_thread(self._client.load_settings, self._session_file)
            await asyncio.to_thread(self._client.get_timeline_feed)  # Validate session
            self._logged_in = True
            logger.info("Instagram session loaded successfully.", extra={"platform": self.platform.value})
            return True
        except Exception:
            logger.warning("Failed to load or validate existing Instagram session.", extra={"platform": self.platform.value})
            return False

    async def _perform_login(self) -> None:
        current_time = time.time()
        if self._login_attempts >= self.settings.INSTAGRAM_MAX_LOGIN_ATTEMPTS:
            if current_time - self._last_login_attempt_time < self.settings.INSTAGRAM_LOGIN_COOLDOWN:
                raise AuthenticationException("Too many failed login attempts. Cooldown active.", self.platform)
            self._login_attempts = 0

        self._login_attempts += 1
        self._last_login_attempt_time = current_time
        
        try:
            logger.info("Attempting new Instagram login...", extra={"platform": self.platform.value})
            await asyncio.to_thread(self._client.login, self._username, self._password)
            await asyncio.to_thread(self._client.dump_settings, self._session_file)
            self._logged_in = True
            self._login_attempts = 0
            logger.info("Instagram login successful and session saved.", extra={"platform": self.platform.value})
        except (TwoFactorRequired, ChallengeRequired) as e:
            raise AuthenticationException(f"Account requires manual intervention: {type(e).__name__}", self.platform)
        except LoginRequired as e:
            raise AuthenticationException(f"Login failed, credentials may be invalid: {e}", self.platform)
        except Exception as e:
            raise ConnectorException(f"An unexpected error occurred during login: {e}", self.platform)

    async def _apply_rate_limiting(self) -> None:
        """Ensure minimum delay between requests, with adaptive backoff on errors."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self._min_delay:
            await asyncio.sleep(self._min_delay - time_since_last)
        self._last_request_time = time.time()
        self._request_count += 1

    async def search(self, query: str) -> List[CreatorProfile]:
        await self._ensure_login()
        clean_query = query.strip().lstrip('@').lower()
        if not clean_query:
            return []

        try:
            await self._apply_rate_limiting()
            logger.debug(f"Searching Instagram for user '{clean_query}'", extra={"platform": self.platform.value})
            user_data = await asyncio.to_thread(self._client.user_info_by_username, clean_query)
            profile = self._map_to_creator_profile(user_data.dict())
            return [profile] if profile else []
        except UserNotFound:
            logger.info(f"Instagram user not found: {clean_query}", extra={"platform": self.platform.value})
            return []
        except PrivateError:
            logger.info(f"Instagram profile is private: {clean_query}", extra={"platform": self.platform.value})
            try:
                # Attempt to return limited info
                basic_info = await asyncio.to_thread(self._client.user_info_by_username, clean_query)
                profile = self._map_to_creator_profile(basic_info.dict(), is_private=True)
                return [profile] if profile else []
            except Exception:
                return []
        except (PleaseWaitFewMinutes, RateLimitError) as e:
            self._min_delay = min(self._min_delay * 2, 10.0)  # exponential backoff
            raise RateLimitException(f"Instagram is rate limiting requests: {e}", self.platform)
        except Exception as e:
            raise ConnectorException(f"Failed to search for user '{clean_query}': {e}", self.platform)

    def _map_to_creator_profile(self, data: Dict[str, Any], is_private: bool = False) -> Optional[CreatorProfile]:
        try:
            username = data.get('username')
            if not username:
                return None
            return CreatorProfile(
                platform=self.platform,
                platform_id=str(data.get('pk')),
                name=data.get('full_name') or username,
                handle=username,
                profile_url=f"https://www.instagram.com/{username}/",
                bio=data.get('biography'),
                avatar_url=data.get('profile_pic_url_hd') or data.get('profile_pic_url'),
                is_verified=data.get('is_verified', False),
                verification_status=VerificationStatus.VERIFIED if data.get('is_verified') else VerificationStatus.UNVERIFIED,
                account_type="business" if data.get('is_business') else "personal",
                is_active=True,
                social_metrics=SocialMetrics(
                    followers_count=data.get('follower_count') if not is_private else None,
                    following_count=data.get('following_count') if not is_private else None,
                    posts_count=data.get('media_count') if not is_private else None,
                    total_content_count=data.get('media_count') if not is_private else None
                ),
                engagement_metrics=EngagementMetrics(),  # Could be extended
                metadata=ProfileMetadata(
                    custom_fields={
                        "website": data.get('external_url'),
                        "business_email": data.get('public_email'),
                        "category": data.get('category'),
                        "is_private": is_private or data.get('is_private', False),
                        "business_account": data.get('is_business', False)
                    }
                ),
                scraped_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Failed to map Instagram data: {e}", extra={"platform": self.platform.value}, exc_info=True)
            return None
    
    async def get_health_status(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "logged_in": self._logged_in,
            "login_attempts": self._login_attempts,
            "last_request_time": self._last_request_time,
            "request_count": self._request_count,
            "rate_limit_delay": self._min_delay,
            "status": "healthy" if self._logged_in else "disconnected"
        }
        
    async def _cleanup(self) -> None:
        if self._logged_in:
            await asyncio.to_thread(self._client.dump_settings, self._session_file)
            logger.info("Instagram session saved during cleanup.", extra={"platform": self.platform.value})