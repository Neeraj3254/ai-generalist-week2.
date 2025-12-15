"""
Production Retry Pattern with Exponential Backoff
Week 2, Day 3, Session 1 - AI Generalist Training

Patterns Implemented:
1. Decorator Pattern: Wrap functions with retry logic
2. Exponential Backoff: Wait longer between retries (1s, 2s, 4s, 8s)
3. Exception Filtering: Only retry specific exceptions
4. Circuit Breaker: Stop retrying if system is down
"""

import time
import logging
import random
from typing import Type, Tuple, Callable, Any
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CORE RETRY DECORATOR
# ============================================================

def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    backoff_multiplier: float = 1.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] = None
):
    """
    Retry decorator with exponential backoff
    
    Production Pattern: Resilient network operations
    
    Args:
        max_attempts: Maximum number of attempts (including initial)
        backoff_base: Base for exponential backoff (default: 2.0)
        backoff_multiplier: Multiplier for backoff time (default: 1.0)
        jitter: Add randomness to prevent thundering herd (default: True)
        exceptions: Tuple of exceptions to catch (default: all exceptions)
        on_retry: Callback function(exception, attempt) called on each retry
    
    Example:
        @retry(max_attempts=5, backoff_base=2.0, exceptions=(requests.RequestException,))
        def fetch_data(url):
            return requests.get(url)
    
    Backoff Formula:
        wait_time = backoff_multiplier * (backoff_base ^ attempt)
        
        With defaults (base=2.0, multiplier=1.0):
        - Attempt 1: 0s (immediate)
        - Attempt 2: 2s = 1.0 * (2^1)
        - Attempt 3: 4s = 1.0 * (2^2)
        - Attempt 4: 8s = 1.0 * (2^3)
        - Attempt 5: 16s = 1.0 * (2^4)
    
    Why Exponential Backoff?
    - Network errors often resolve themselves (transient failures)
    - Waiting longer gives system time to recover
    - Prevents overwhelming a struggling server
    
    Why Jitter?
    - Prevents "thundering herd" (all clients retry at exact same time)
    - Adds randomness: wait_time ± 25%
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)  # Preserve original function metadata
        def wrapper(*args, **kwargs) -> Any:
            """
            Wrapper function that implements retry logic
            
            Production Pattern: Separate concerns
            - Wrapper handles retries
            - Original function handles business logic
            """
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    
                    # Success - log if this wasn't first attempt
                    if attempt > 1:
                        logger.info(
                            f"✓ {func.__name__} succeeded on attempt {attempt}/{max_attempts}"
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Log the failure
                    logger.warning(
                        f"✗ {func.__name__} failed on attempt {attempt}/{max_attempts}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)
                    
                    # If this was the last attempt, don't sleep
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts"
                        )
                        break
                    
                    # Calculate backoff time
                    wait_time = backoff_multiplier * (backoff_base ** (attempt - 1))
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        # Add random ±25% variation
                        jitter_range = wait_time * 0.25
                        wait_time = wait_time + random.uniform(-jitter_range, jitter_range)
                    
                    # Ensure minimum wait time
                    wait_time = max(0.1, wait_time)
                    
                    logger.info(f"⏳ Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
            
            # All attempts failed - raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================
# SPECIALIZED RETRY DECORATORS
# ============================================================

def retry_network(max_attempts: int = 3):
    """
    Specialized decorator for network operations
    
    Production Pattern: Domain-specific retries
    Why: Different error types need different retry strategies
    """
    import requests
    
    return retry(
        max_attempts=max_attempts,
        backoff_base=2.0,
        exceptions=(
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ConnectionError,
            TimeoutError
        )
    )


def retry_database(max_attempts: int = 5):
    """
    Specialized decorator for database operations
    
    Production Pattern: Longer retries for DB connection issues
    """
    return retry(
        max_attempts=max_attempts,
        backoff_base=1.5,
        backoff_multiplier=0.5,
        exceptions=(ConnectionError, TimeoutError)
    )


def retry_api(max_attempts: int = 3, rate_limit_wait: float = 60.0):
    """
    Specialized decorator for API calls
    
    Production Pattern: Handle rate limiting specifically
    """
    def on_retry_callback(exception: Exception, attempt: int):
        # If rate limited, wait longer
        if "429" in str(exception) or "rate limit" in str(exception).lower():
            logger.warning(f"Rate limit hit - waiting {rate_limit_wait}s")
            time.sleep(rate_limit_wait)
    
    return retry(
        max_attempts=max_attempts,
        backoff_base=2.0,
        on_retry=on_retry_callback
    )


# ============================================================
# CIRCUIT BREAKER PATTERN
# ============================================================

class CircuitBreaker:
    """
    Circuit Breaker: Stop retrying if system is clearly down
    
    Production Pattern: Fail fast when service is unavailable
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    
    Why:
    - Prevents wasting time on dead services
    - Gives failing service time to recover
    - Protects against cascading failures
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Production Pattern: Wraps calls with circuit breaker logic
        """
        # Check if circuit is OPEN
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker entering HALF_OPEN state (testing recovery)")
                self.state = "HALF_OPEN"
            else:
                raise Exception(
                    f"Circuit breaker OPEN - service unavailable "
                    f"(retry in {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s)"
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker closing (service recovered)")
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Check if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker OPENING ({self.failure_count} failures)"
                )
                self.state = "OPEN"
            
            raise e


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import requests
    
    print("=" * 60)
    print("RETRY PATTERN DEMONSTRATIONS")
    print("=" * 60)
    
    # Example 1: Basic retry with simulated failures
    print("\n1. BASIC RETRY WITH SIMULATED FAILURES")
    
    attempt_counter = [0]  # Mutable to track across closure
    
    @retry(max_attempts=4, backoff_base=2.0)
    def flaky_function():
        """Simulates a function that fails first 2 times"""
        attempt_counter[0] += 1
        if attempt_counter[0] < 3:
            raise ConnectionError(f"Simulated failure #{attempt_counter[0]}")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Example 2: Network retry with real HTTP request
    print("\n2. NETWORK RETRY (Real HTTP Request)")
    
    @retry_network(max_attempts=3)
    def fetch_url(url: str):
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return f"Fetched {len(response.text)} bytes"
    
    try:
        result = fetch_url("https://httpbin.org/status/200")
        print(f"✓ {result}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Example 3: Retry with callback
    print("\n3. RETRY WITH CUSTOM CALLBACK")
    
    retry_log = []
    
    def log_retry(exception: Exception, attempt: int):
        retry_log.append(f"Attempt {attempt}: {type(exception).__name__}")
    
    @retry(max_attempts=3, on_retry=log_retry)
    def failing_function():
        raise ValueError("Always fails")
    
    try:
        failing_function()
    except ValueError:
        print("Function failed as expected")
        print(f"Retry log: {retry_log}")
    
    # Example 4: Circuit Breaker
    print("\n4. CIRCUIT BREAKER PATTERN")
    
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)
    
    def unstable_service():
        raise ConnectionError("Service down")
    
    # Trigger circuit breaker
    for i in range(5):
        try:
            breaker.call(unstable_service)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
        time.sleep(0.5)
    
    print("\n5. COMPARISON: With vs Without Retry")
    
    # Without retry (fails immediately)
    def fetch_without_retry():
        response = requests.get("https://httpbin.org/status/500")
        response.raise_for_status()
    
    # With retry (retries 3 times)
    @retry_network(max_attempts=3)
    def fetch_with_retry():
        response = requests.get("https://httpbin.org/status/500")
        response.raise_for_status()
    
    start = time.time()
    try:
        fetch_without_retry()
    except Exception:
        elapsed_no_retry = time.time() - start
        print(f"Without retry: Failed in {elapsed_no_retry:.2f}s")
    
    start = time.time()
    try:
        fetch_with_retry()
    except Exception:
        elapsed_with_retry = time.time() - start
        print(f"With retry: Failed in {elapsed_with_retry:.2f}s (gave service {elapsed_with_retry/elapsed_no_retry:.1f}x more chances)")