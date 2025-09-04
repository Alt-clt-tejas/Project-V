# app/exceptions.py
class ConnectorException(Exception):
    """Base exception for connector errors."""
    def __init__(self, message: str, platform: str = None, recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.platform = platform
        self.recoverable = recoverable

class RateLimitException(ConnectorException):
    """Exception representing rate limit scenarios."""
    def __init__(self, message: str, platform: str = None, retry_after: int = None):
        super().__init__(message, platform, recoverable=True)
        self.retry_after = retry_after

class AuthenticationException(ConnectorException):
    """Exception representing authentication failures."""
    def __init__(self, message: str, platform: str = None):
        super().__init__(message, platform, recoverable=False)