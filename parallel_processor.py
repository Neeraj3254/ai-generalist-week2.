"""
Production Parallel Processing Engine
Week 2, Day 3, Session 2 - AI Generalist Training

Patterns Implemented:
1. ThreadPoolExecutor: Parallel I/O operations
2. Future tracking: Monitor individual task progress
3. Graceful degradation: One failure doesn't kill all tasks
4. Progress reporting: Real-time completion tracking
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import requests

from retry_pattern import retry_network

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TaskResult:
    """Result of a single task execution"""
    task_id: str
    success: bool
    data: Any = None
    error: str = None
    duration: float = 0.0
    attempts: int = 1


@dataclass
class BatchResult:
    """Result of batch processing"""
    total_tasks: int
    successful: int = 0
    failed: int = 0
    results: List[TaskResult] = field(default_factory=list)
    duration: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.successful / self.total_tasks) * 100
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"Batch Processing Summary:\n"
            f"  Total: {self.total_tasks}\n"
            f"  ✓ Successful: {self.successful} ({self.success_rate:.1f}%)\n"
            f"  ✗ Failed: {self.failed}\n"
            f"  Duration: {self.duration:.2f}s\n"
            f"  Throughput: {self.total_tasks/self.duration:.2f} tasks/sec"
        )


# ============================================================
# PARALLEL PROCESSOR
# ============================================================

class ParallelProcessor:
    """
    Production-grade parallel processing engine
    
    Features:
    - ThreadPoolExecutor for I/O bound tasks
    - Individual task error handling
    - Progress tracking
    - Retry integration
    - Timeout handling
    
    Why ThreadPoolExecutor?
    - Clean API for parallel execution
    - Built-in thread management
    - Exception handling per task
    - Integrates with context managers (auto-cleanup)
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        timeout: float = 30.0,
        show_progress: bool = True
    ):
        """
        Args:
            max_workers: Maximum concurrent threads
            timeout: Timeout per task (seconds)
            show_progress: Print progress updates
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.show_progress = show_progress
        
        logger.info(f"ParallelProcessor initialized with {max_workers} workers")
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        task_id_func: Callable[[Any], str] = None
    ) -> BatchResult:
        """
        Process a batch of items in parallel
        
        Production Pattern: Map-reduce for parallel processing
        
        Args:
            items: List of items to process
            process_func: Function to process each item (should be thread-safe)
            task_id_func: Function to extract task ID from item (default: str(item))
        
        Returns:
            BatchResult with all task results
        
        Example:
            processor = ParallelProcessor(max_workers=5)
            
            def fetch_url(url):
                response = requests.get(url)
                return response.text
            
            result = processor.process_batch(urls, fetch_url)
        """
        if not items:
            return BatchResult(total_tasks=0)
        
        # Task ID function
        if task_id_func is None:
            task_id_func = lambda x: str(x)
        
        batch_result = BatchResult(total_tasks=len(items))
        start_time = time.time()
        
        logger.info(f"Starting batch processing: {len(items)} items with {self.max_workers} workers")
        
        # Create thread pool and submit all tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and create future -> item mapping
            future_to_item = {
                executor.submit(self._execute_task, process_func, item, task_id_func(item)): item
                for item in items
            }
            
            # Process completed tasks as they finish
            completed = 0
            for future in as_completed(future_to_item):
                completed += 1
                
                # Get task result
                task_result = future.result()
                batch_result.results.append(task_result)
                
                # Update counters
                if task_result.success:
                    batch_result.successful += 1
                else:
                    batch_result.failed += 1
                
                # Progress logging
                if self.show_progress:
                    status = "✓" if task_result.success else "✗"
                    logger.info(
                        f"[{completed}/{len(items)}] {status} {task_result.task_id} "
                        f"({task_result.duration:.2f}s)"
                    )
                    if not task_result.success:
                        logger.warning(f"  Error: {task_result.error}")
        
        batch_result.duration = time.time() - start_time
        
        logger.info("\n" + batch_result.summary())
        
        return batch_result
    
    def _execute_task(
        self,
        func: Callable,
        item: Any,
        task_id: str
    ) -> TaskResult:
        """
        Execute a single task with error handling and timing
        
        Production Pattern: Isolate failures
        Why: One task failure shouldn't affect others
        """
        start_time = time.time()
        
        try:
            # Execute the function
            result = func(item)
            
            return TaskResult(
                task_id=task_id,
                success=True,
                data=result,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            # Task failed - log and return error result
            logger.debug(f"Task {task_id} failed: {type(e).__name__}: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration=time.time() - start_time
            )


# ============================================================
# SPECIALIZED PROCESSORS
# ============================================================

class ParallelScraper(ParallelProcessor):
    """
    Specialized parallel processor for web scraping
    
    Production Pattern: Domain-specific processing
    Why: Scraping has specific needs (retry, rate limiting, headers)
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        timeout: float = 30.0,
        user_agent: str = "ParallelScraper/1.0"
    ):
        super().__init__(max_workers, timeout)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
    
    @retry_network(max_attempts=3)
    def _fetch_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch a single URL with retry logic
        
        Production Pattern: Retry at the lowest level
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return {
            'url': url,
            'status_code': response.status_code,
            'content_length': len(response.content),
            'headers': dict(response.headers)
        }
    
    def scrape_urls(self, urls: List[str]) -> BatchResult:
        """
        Scrape multiple URLs in parallel
        
        Production API: High-level method for common use case
        """
        return self.process_batch(
            items=urls,
            process_func=self._fetch_url,
            task_id_func=lambda url: url
        )


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def compare_serial_vs_parallel():
    """
    Demonstrate speed improvement of parallel processing
    
    Production Insight: Measure before optimizing
    """
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1"
    ]
    
    print("\n" + "=" * 60)
    print("SERIAL VS PARALLEL COMPARISON")
    print("=" * 60)
    
    # Serial processing
    print("\n1. SERIAL PROCESSING (one at a time)")
    start = time.time()
    
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ Fetched {url}")
        except Exception as e:
            print(f"✗ Failed {url}: {e}")
    
    serial_time = time.time() - start
    print(f"\nSerial time: {serial_time:.2f}s")
    
    # Parallel processing
    print("\n2. PARALLEL PROCESSING (5 concurrent)")
    scraper = ParallelScraper(max_workers=5)
    
    start = time.time()
    result = scraper.scrape_urls(urls)
    parallel_time = time.time() - start
    
    print(f"\nParallel time: {parallel_time:.2f}s")
    print(f"Speedup: {serial_time/parallel_time:.2f}x faster")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL PROCESSING DEMONSTRATIONS")
    print("=" * 60)
    
    # Example 1: Basic parallel processing
    print("\n1. BASIC PARALLEL PROCESSING")
    
    def slow_function(n: int) -> int:
        """Simulate slow computation"""
        time.sleep(0.5)
        return n * n
    
    processor = ParallelProcessor(max_workers=5, show_progress=True)
    numbers = list(range(10))
    
    result = processor.process_batch(
        items=numbers,
        process_func=slow_function,
        task_id_func=lambda n: f"task_{n}"
    )
    
    print(f"\nResults: {[r.data for r in result.results if r.success]}")
    
    # Example 2: Parallel web scraping
    print("\n2. PARALLEL WEB SCRAPING")
    
    test_urls = [
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",  # Will fail
        "https://httpbin.org/status/200",
        "https://invalid-url-12345.com",  # Will fail
    ]
    
    scraper = ParallelScraper(max_workers=3)
    result = scraper.scrape_urls(test_urls)
    
    # Show successful results
    print("\nSuccessful fetches:")
    for task_result in result.results:
        if task_result.success:
            print(f"  ✓ {task_result.task_id}: {task_result.data['content_length']} bytes")
    
    # Show failures
    print("\nFailed fetches:")
    for task_result in result.results:
        if not task_result.success:
            print(f"  ✗ {task_result.task_id}: {task_result.error}")
    
    # Example 3: Performance comparison
    compare_serial_vs_parallel()
    
    # Example 4: Error handling
    print("\n3. GRACEFUL ERROR HANDLING")
    
    def sometimes_fails(n: int) -> int:
        """Function that fails randomly"""
        if n % 3 == 0:
            raise ValueError(f"Number {n} is divisible by 3!")
        return n * 2
    
    result = processor.process_batch(
        items=list(range(10)),
        process_func=sometimes_fails,
        task_id_func=lambda n: f"num_{n}"
    )
    
    print(f"\nProcessed {result.total_tasks} items:")
    print(f"  ✓ Success: {result.successful}")
    print(f"  ✗ Failed: {result.failed}")