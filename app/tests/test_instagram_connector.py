# tests/test_instagram_connector.py - Comprehensive test suite
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from app.connectors.instagram_connector import (
    InstagramConnector, InstagramConnectionError, InstagramRateLimitError
)
from app.config.base import AppSettings
from app.domains.search.schemas import Platform, CreatorProfile

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=AppSettings)
    settings.INSTAGRAM_USERNAME = Mock()
    settings.INSTAGRAM_USERNAME.get_secret_value.return_value = "test_user"
    settings.INSTAGRAM_PASSWORD = Mock()
    settings.INSTAGRAM_PASSWORD.get_secret_value.return_value = "test_pass"
    settings.INSTAGRAM_SESSION_PATH = "./test_sessions"
    settings.INSTAGRAM_RATE_LIMIT_DELAY = 1.0
    settings.INSTAGRAM_MAX_LOGIN_ATTEMPTS = 3
    settings.INSTAGRAM_LOGIN_COOLDOWN = 300
    return settings

@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    return AsyncMock()

@pytest.fixture
def sample_instagram_data():
    """Sample Instagram user data for testing."""
    return {
        'pk': '123456789',
        'username': 'testuser',
        'full_name': 'Test User',
        'biography': 'Test bio',
        'follower_count': 1000,
        'following_count': 500,
        'media_count': 50,
        'is_verified': True,
        'is_business': False,
        'is_private': False,
        'profile_pic_url_hd': 'https://example.com/pic.jpg',
        'external_url': 'https://example.com',
        'public_email': 'test@example.com'
    }

class TestInstagramConnector:
    """Test suite for Instagram connector."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_settings, mock_http_client):
        """Test successful connector initialization."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        assert connector.platform == Platform.INSTAGRAM
        assert not connector._logged_in
    
    @pytest.mark.asyncio
    async def test_initialization_missing_credentials(self, mock_http_client):
        """Test initialization with missing credentials."""
        settings = Mock(spec=AppSettings)
        settings.INSTAGRAM_USERNAME = None
        settings.INSTAGRAM_PASSWORD = None
        
        with pytest.raises(ValueError, match="INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD"):
            InstagramConnector(settings, mock_http_client)
    
    @pytest.mark.asyncio
    async def test_login_success(self, mock_settings, mock_http_client):
        """Test successful login."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        
        with patch.object(connector._client, 'login') as mock_login:
            with patch('asyncio.to_thread', return_value=None) as mock_thread:
                await connector._ensure_login()
                assert connector._logged_in
                mock_thread.assert_called()
    
    @pytest.mark.asyncio
    async def test_login_failure(self, mock_settings, mock_http_client):
        """Test login failure handling."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        
        with patch('asyncio.to_thread', side_effect=Exception("Login failed")):
            with pytest.raises(InstagramConnectionError):
                await connector._ensure_login()
            assert not connector._logged_in
    
    @pytest.mark.asyncio
    async def test_search_success(self, mock_settings, mock_http_client, sample_instagram_data):
        """Test successful user search."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        connector._logged_in = True
        
        mock_user_data = Mock()
        mock_user_data.dict.return_value = sample_instagram_data
        
        with patch('asyncio.to_thread', return_value=mock_user_data):
            results = await connector.search("testuser")
            
            assert len(results) == 1
            profile = results[0]
            assert isinstance(profile, CreatorProfile)
            assert profile.handle == "testuser"
            assert profile.platform == Platform.INSTAGRAM
            assert profile.is_verified == True
    
    @pytest.mark.asyncio
    async def test_search_user_not_found(self, mock_settings, mock_http_client):
        """Test search when user is not found."""
        from instagrapi.exceptions import UserNotFound
        
        connector = InstagramConnector(mock_settings, mock_http_client)
        connector._logged_in = True
        
        with patch('asyncio.to_thread', side_effect=UserNotFound("User not found")):
            results = await connector.search("nonexistentuser")
            assert results == []
    
    @pytest.mark.asyncio
    async def test_search_private_profile(self, mock_settings, mock_http_client):
        """Test search for private profile."""
        from instagrapi.exceptions import PrivateError
        
        connector = InstagramConnector(mock_settings, mock_http_client)
        connector._logged_in = True
        
        with patch('asyncio.to_thread', side_effect=PrivateError("Private profile")):
            results = await connector.search("privateuser")
            assert results == []
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_settings, mock_http_client):
        """Test rate limiting functionality."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        
        # First call should set the timestamp
        await connector._apply_rate_limiting()
        first_time = connector._rate_limit.last_request_time
        
        # Immediate second call should cause delay
        with patch('asyncio.sleep') as mock_sleep:
            await connector._apply_rate_limiting()
            mock_sleep.assert_called_once()
        
        assert connector._rate_limit.last_request_time > first_time
    
    @pytest.mark.asyncio
    async def test_health_status(self, mock_settings, mock_http_client):
        """Test health status reporting."""
        connector = InstagramConnector(mock_settings, mock_http_client)
        connector._logged_in = True
        
        health = await connector.get_health_status()
        
        assert health["platform"] == Platform.INSTAGRAM.value
        assert health["logged_in"] == True
        assert health["status"] == "healthy"
        assert "request_count" in health
        assert "rate_limit_delay" in health

# tests/conftest.py - Test configuration
import pytest
import asyncio
from unittest.mock import AsyncMock
import httpx

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def http_client():
    """Create a test HTTP client."""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.fixture
def mock_async_client():
    """Create a mock async HTTP client."""
    return AsyncMock(spec=httpx.AsyncClient)

# Docker configuration
"""
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs sessions

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

"""
# docker-compose.yml
version: '3.8'

services:
  creator-search-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./sessions:/app/sessions
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
  # Optional: Redis for caching (future enhancement)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  redis_data:
"""

# pyproject.toml - Poetry configuration
"""
[tool.poetry]
name = "creator-search-api"
version = "1.0.0"
description = "Professional creator search API with multi-platform support"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
httpx = "^0.25.2"
pydantic = {extras = ["email"], version = "^2.5.0"}
instagrapi = "^2.0.0"
python-multipart = "^0.0.6"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
pytest-mock = "^3.12.0"
black = "^23.11.0"
isort = "^5.12.0"
flake8 = "^6.1.0"
mypy = "^1.7.1"
pre-commit = "^3.6.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
"""

# Makefile for common tasks
"""
.PHONY: install test lint format run docker-build docker-run clean help

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using Poetry
	poetry install

test: ## Run tests
	poetry run pytest -v --tb=short

test-cov: ## Run tests with coverage
	poetry run pytest --cov=app --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	poetry run flake8 app tests
	poetry run mypy app

format: ## Format code using black and isort
	poetry run black app tests
	poetry run isort app tests

format-check: ## Check code formatting
	poetry run black --check app tests
	poetry run isort --check-only app tests

run: ## Run the development server
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the production server
	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t creator-search-api .

docker-run: ## Run Docker container
	docker-compose up -d

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-stop: ## Stop Docker containers
	docker-compose down

clean: ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf dist build *.egg-info

setup-dev: install ## Setup development environment
	poetry run pre-commit install
	mkdir -p logs sessions
	cp .env.example .env
	@echo "Development environment setup complete!"
	@echo "Please edit .env file with your API credentials."

health-check: ## Check API health
	curl -f http://localhost:8000/api/v1/health || echo "API not running"

test-instagram: ## Test Instagram search
	curl -X POST "http://localhost:8000/api/v1/search" \
		-H "Content-Type: application/json" \
		-d '{"query": "mkbhd", "search_type": "creator", "filters": {"platforms": ["Instagram"]}}'
"""