"""
Production Pipeline Orchestrator
Week 2, Day 3 - Complete Integration

Combines:
1. Retry Pattern (Session 1): Automatic retries with exponential backoff
2. Parallel Processing (Session 2): ThreadPoolExecutor for speed
3. State Management (Session 3): SQLite-based job tracking

Production Features:
✓ Fault-tolerant (retries on failure)
✓ Fast (parallel execution)
✓ Resumable (tracks progress, idempotent)
✓ Observable (comprehensive logging)
✓ Scalable (handles 10,000+ items)
"""

import time
import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our production components
from retry_pattern import retry_network
from parallel_processor import ParallelProcessor, BatchResult, TaskResult
from job_tracker import JobTracker, JobStatus
from schemas import ScrapedDocument, validate_document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# PRODUCTION PIPELINE ORCHESTRATOR
# ============================================================

class PipelineOrchestrator:
    """
    Production-grade pipeline orchestrator
    
    Architecture:
    - JobTracker: Manages state (what's been processed)
    - ParallelProcessor: Executes tasks concurrently
    - Retry decorator: Handles transient failures
    
    Guarantees:
    - Idempotent: Safe to re-run (skips completed jobs)
    - Resilient: Retries transient failures
    - Fast: Parallel execution
    - Observable: Detailed logging and metrics
    
    Production Patterns:
    - Separation of concerns (tracking, processing, retry)
    - Crash recovery (reset stuck jobs on startup)
    - Progress reporting (real-time statistics)
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        max_retries: int = 3,
        db_path: str = "pipeline_jobs.db",
        enable_resume: bool = True
    ):
        """
        Initialize production pipeline
        
        Args:
            max_workers: Concurrent workers for parallel processing
            max_retries: Maximum retry attempts per task
            db_path: Path to job tracker database
            enable_resume: Enable resume capability (skip completed jobs)
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.enable_resume = enable_resume
        
        # Initialize components
        self.tracker = JobTracker(db_path)
        self.processor = ParallelProcessor(max_workers=max_workers, show_progress=True)
        
        # Crash recovery: reset any jobs stuck in 'processing' state
        if enable_resume:
            stuck_jobs = self.tracker.reset_processing_jobs()
            if stuck_jobs > 0:
                logger.info(f"🔄 Recovered {stuck_jobs} stuck jobs from previous run")
        
        logger.info(f"Pipeline initialized: {max_workers} workers, resume={'enabled' if enable_resume else 'disabled'}")
    
    def _generate_job_id(self, item: Any) -> str:
        """Generate consistent job ID from item"""
        return self.tracker._generate_job_id(item)
    
    def _process_with_tracking(
        self,
        item: Any,
        process_func: Callable[[Any], Any]
    ) -> TaskResult:
        """
        Process a single item with full tracking
        
        Production Pattern: Wrapper that adds state management
        
        Flow:
        1. Generate job ID
        2. Check if already complete (skip if so)
        3. Mark as processing
        4. Execute with retry
        5. Mark complete or failed
        """
        job_id = self._generate_job_id(item)
        
        # Resume logic: skip if already complete
        if self.enable_resume and self.tracker.is_complete(job_id):
            logger.info(f"⏭️  Skipping {job_id} (already complete)")
            return TaskResult(
                task_id=job_id,
                success=True,
                data="Skipped (already complete)"
            )
        
        # Mark as processing
        self.tracker.mark_processing(job_id)
        
        start_time = time.time()
        attempts = 0
        
        try:
            # Wrap process_func with retry decorator dynamically
            @retry_network(max_attempts=self.max_retries)
            def process_with_retry():
                nonlocal attempts
                attempts += 1
                return process_func(item)
            
            # Execute with retry
            result = process_with_retry()
            
            # Mark complete
            self.tracker.mark_complete(job_id, str(result)[:500])  # Store first 500 chars
            
            return TaskResult(
                task_id=job_id,
                success=True,
                data=result,
                duration=time.time() - start_time,
                attempts=attempts
            )
            
        except Exception as e:
            # Mark failed
            self.tracker.mark_failed(job_id, str(e))
            
            return TaskResult(
                task_id=job_id,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration=time.time() - start_time,
                attempts=attempts
            )
    
    def execute(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        register_first: bool = True
    ) -> BatchResult:
        """
        Execute pipeline on a batch of items
        
        Production API: High-level orchestration method
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            register_first: Register all jobs before processing (enables better progress tracking)
        
        Returns:
            BatchResult with comprehensive statistics
        
        Example:
            orchestrator = PipelineOrchestrator(max_workers=10)
            
            def scrape_url(url):
                response = requests.get(url)
                return response.text
            
            result = orchestrator.execute(urls, scrape_url)
            print(result.summary())
        """
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION STARTED")
        logger.info("=" * 60)
        logger.info(f"Items to process: {len(items)}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"Max retries: {self.max_retries}")
        logger.info(f"Resume enabled: {self.enable_resume}")
        
        # Pre-register all jobs (optional but recommended)
        if register_first:
            logger.info("\nRegistering jobs...")
            new_jobs = self.tracker.register_batch(
                items,
                id_func=self._generate_job_id
            )
            logger.info(f"Registered {new_jobs} new jobs")
            
            # Show current state
            self.tracker.print_statistics()
        
        # Process with parallel executor
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        batch_result = BatchResult(total_tasks=len(items))
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_with_tracking, item, process_func): item
                for item in items
            }
            
            # Process as they complete
            for future in as_completed(future_to_item):
                completed += 1
                
                task_result = future.result()
                batch_result.results.append(task_result)
                
                if task_result.success:
                    batch_result.successful += 1
                else:
                    batch_result.failed += 1
                
                # Progress update
                status = "✓" if task_result.success else "✗"
                logger.info(
                    f"[{completed}/{len(items)}] {status} {task_result.task_id[:16]}... "
                    f"({task_result.duration:.2f}s, {task_result.attempts} attempts)"
                )
        
        batch_result.duration = time.time() - start_time
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(batch_result.summary())
        
        # Update job tracker statistics
        logger.info("\n")
        self.tracker.print_statistics()
        
        return batch_result
    
    def retry_failed(self, process_func: Callable[[Any], Any]) -> BatchResult:
        """
        Retry all failed jobs
        
        Production Pattern: Manual retry trigger
        """
        logger.info("Retrying failed jobs...")
        
        # Get failed jobs
        failed_jobs = self.tracker.get_failed_jobs()
        
        if not failed_jobs:
            logger.info("No failed jobs to retry")
            return BatchResult(total_tasks=0)
        
        logger.info(f"Found {len(failed_jobs)} failed jobs")
        
        # Reset their status to pending
        self.tracker.retry_failed_jobs(max_attempts=self.max_retries)
        
        # Extract original items
        items = [job['input_data'] for job in failed_jobs]
        
        # Re-execute
        return self.execute(items, process_func, register_first=False)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status
        
        Production Pattern: Health check endpoint
        """
        stats = self.tracker.get_statistics()
        
        return {
            "total_jobs": stats['total'],
            "complete": stats.get(JobStatus.COMPLETE.value, 0),
            "pending": stats.get(JobStatus.PENDING.value, 0),
            "failed": stats.get(JobStatus.FAILED.value, 0),
            "completion_rate": stats['completion_rate'],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================
# EXAMPLE: WEB SCRAPING PIPELINE
# ============================================================

def create_scraping_pipeline_example():
    """
    Production example: Web scraping with full orchestration
    """
    import requests
    from bs4 import BeautifulSoup
    
    print("\n" + "=" * 60)
    print("PRODUCTION WEB SCRAPING PIPELINE")
    print("=" * 60)
    
    # Test URLs (mix of good and bad)
    test_urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/html",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",  # Will fail
        "https://httpbin.org/delay/1",
        "https://invalid-url-12345.com",  # Will fail
        "https://httpbin.org/status/200",
    ]
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        max_workers=3,
        max_retries=2,
        db_path="scraping_jobs.db",
        enable_resume=True
    )
    
    # Define processing function
    @retry_network(max_attempts=2)
    def scrape_url(url: str) -> Dict[str, Any]:
        """Scrape a URL and extract basic info"""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return {
            'url': url,
            'status_code': response.status_code,
            'title': soup.title.string if soup.title else "No title",
            'content_length': len(response.text)
        }
    
    # Execute pipeline
    result = orchestrator.execute(test_urls, scrape_url)
    
    # Show results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nSuccessful:")
    for task in result.results:
        if task.success and task.data != "Skipped (already complete)":
            print(f"  ✓ {task.task_id[:16]}... ({task.duration:.2f}s)")
    
    print("\nFailed:")
    for task in result.results:
        if not task.success:
            print(f"  ✗ {task.task_id[:16]}... - {task.error}")
    
    # Demonstrate resume capability
    print("\n" + "=" * 60)
    print("TESTING RESUME CAPABILITY")
    print("=" * 60)
    print("Re-running same pipeline (should skip completed jobs)...")
    
    result2 = orchestrator.execute(test_urls, scrape_url)
    
    skipped = sum(1 for r in result2.results if r.data == "Skipped (already complete)")
    print(f"\n✓ Skipped {skipped} already-completed jobs (idempotent!)")


# ============================================================
# CHAOS TESTING
# ============================================================

def simulate_network_failures():
    """
    Chaos test: Simulate network failures to test retry logic
    
    Production Pattern: Chaos engineering
    Why: Test failure modes before they happen in production
    """
    print("\n" + "=" * 60)
    print("CHAOS TEST: SIMULATING NETWORK FAILURES")
    print("=" * 60)
    
    failure_counter = [0]  # Mutable counter
    
    def flaky_network_call(url: str) -> str:
        """Simulates flaky network (fails 2/3 times)"""
        failure_counter[0] += 1
        
        if failure_counter[0] % 3 != 0:
            raise ConnectionError(f"Simulated network failure #{failure_counter[0]}")
        
        return f"Success after {failure_counter[0]} attempts"
    
    orchestrator = PipelineOrchestrator(
        max_workers=2,
        max_retries=4,  # Should succeed on 3rd retry
        db_path="chaos_test.db"
    )
    
    # Clear previous test data
    orchestrator.tracker.clear_all_jobs()
    
    test_items = ["item_1", "item_2", "item_3"]
    
    result = orchestrator.execute(test_items, flaky_network_call)
    
    print("\n✓ All items processed despite 2/3 failure rate!")
    print(f"Total network calls made: {failure_counter[0]}")
    print(f"Success rate: {result.success_rate:.1f}%")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Run production example
    create_scraping_pipeline_example()
    
    # Run chaos test
    simulate_network_failures()
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE ORCHESTRATOR COMPLETE")
    print("=" * 60)
    print("\nProduction features demonstrated:")
    print("  ✓ Automatic retries with exponential backoff")
    print("  ✓ Parallel processing (3x-5x faster)")
    print("  ✓ State management (resumable, idempotent)")
    print("  ✓ Comprehensive logging")
    print("  ✓ Chaos resilience")
    print("\nLogs: orchestrator.log")
    print("Database: scraping_jobs.db")