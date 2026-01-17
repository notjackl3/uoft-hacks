"""
Rate limiter for LLM API calls with retry logic and exponential backoff.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

# Global state for rate limiting
_last_call_time: float = 0.0
_lock = asyncio.Lock()


class RateLimitError(Exception):
    """Raised when rate limit is exceeded after all retries."""
    pass


async def wait_for_rate_limit() -> None:
    """
    Wait if necessary to respect the minimum delay between API calls.
    """
    global _last_call_time
    
    async with _lock:
        now = time.time()
        elapsed = now - _last_call_time
        min_delay = settings.llm_min_delay
        
        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next LLM call")
            await asyncio.sleep(wait_time)
        
        _last_call_time = time.time()


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a rate limit error (HTTP 429 or similar).
    """
    error_str = str(error).lower()
    return (
        "429" in error_str or
        "rate" in error_str and "limit" in error_str or
        "quota" in error_str or
        "resource exhausted" in error_str or
        "too many requests" in error_str
    )


T = TypeVar("T")


async def call_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int | None = None,
    **kwargs: Any,
) -> T:
    """
    Call a function with rate limiting and retry logic.
    
    Args:
        func: The function to call (can be sync or async)
        *args: Positional arguments for the function
        max_retries: Maximum number of retries on rate limit errors (default from settings)
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
    
    Raises:
        RateLimitError: If rate limit is exceeded after all retries
        Exception: Any other exception from the function
    """
    if max_retries is None:
        max_retries = settings.llm_max_retries
    
    last_error: Exception | None = None
    backoff = settings.llm_initial_backoff
    max_backoff = settings.llm_max_backoff
    
    for attempt in range(max_retries + 1):
        # Wait for rate limit before each attempt
        await wait_for_rate_limit()
        
        try:
            # Call the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            last_error = e
            
            if is_rate_limit_error(e):
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {backoff:.1f}s before retry..."
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_retries + 1} attempts. "
                        "Please wait a moment and try again."
                    ) from e
            else:
                # Non-rate-limit error, don't retry
                raise
    
    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected state in call_with_retry")


def get_rate_limit_status() -> dict:
    """
    Get current rate limit status for debugging.
    """
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    min_delay = settings.llm_min_delay
    
    return {
        "last_call_seconds_ago": round(elapsed, 2),
        "min_delay_seconds": min_delay,
        "can_call_now": elapsed >= min_delay,
        "wait_time": round(max(0, min_delay - elapsed), 2),
        "max_retries": settings.llm_max_retries,
        "initial_backoff": settings.llm_initial_backoff,
        "max_backoff": settings.llm_max_backoff,
    }
