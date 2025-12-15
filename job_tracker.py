"""
Production Job Tracker with State Management
Week 2, Day 3, Session 3 - AI Generalist Training

Features:
1. Persistent state (SQLite)
2. Job status tracking (pending, processing, complete, failed)
3. Resume capability (skip completed jobs)
4. Thread-safe operations
5. Statistics and reporting
"""

import sqlite3
import threading
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


# ============================================================
# JOB STATUS ENUM
# ============================================================

class JobStatus(str, Enum):
    """Job processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


# ============================================================
# JOB TRACKER
# ============================================================

class JobTracker:
    """
    Production-grade job tracking system
    
    Features:
    - Persistent state (survives crashes)
    - Idempotent (safe to re-run)
    - Thread-safe (for parallel processing)
    - Queryable (statistics, failed jobs, etc.)
    
    Database Schema:
        jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT,
            input_data TEXT,
            result_data TEXT,
            error TEXT,
            attempts INT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    
    Why SQLite?
    - Serverless (no external DB needed)
    - ACID compliant (crash-safe)
    - Fast for local processing
    - SQL queries for analytics
    """
    
    def __init__(self, db_path: str = "job_tracker.db"):
        """
        Initialize job tracker with SQLite database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage for connections
        
        # Initialize database
        self._init_database()
        
        logger.info(f"JobTracker initialized (database: {db_path})")
    
    @contextmanager
    def _get_connection(self):
        """
        Get thread-local database connection
        
        Production Pattern: Thread-local connections
        Why: SQLite connections are not thread-safe
        
        Using context manager ensures connections are properly closed
        """
        # Create connection if not exists for this thread
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0  # Wait up to 30s for lock
            )
            self.local.conn.row_factory = sqlite3.Row  # Dict-like rows
        
        try:
            yield self.local.conn
        except Exception:
            self.local.conn.rollback()
            raise
        else:
            self.local.conn.commit()
    
    def _init_database(self):
        """
        Create database schema if not exists
        
        Production Pattern: Idempotent schema creation
        """
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_data TEXT,
                    result_data TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON jobs(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at 
                ON jobs(updated_at)
            """)
    
    def _generate_job_id(self, input_data: Any) -> str:
        """
        Generate unique job ID from input data
        
        Production Pattern: Content-based ID
        Why: Same input = same ID = idempotent
        """
        content = str(input_data)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    # ========================================
    # JOB REGISTRATION
    # ========================================
    
    def register_job(
        self,
        job_id: str,
        input_data: str,
        status: JobStatus = JobStatus.PENDING
    ) -> bool:
        """
        Register a new job (or skip if exists)
        
        Returns: True if job was created, False if already exists
        
        Production Pattern: INSERT OR IGNORE
        Why: Idempotent registration
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO jobs (job_id, status, input_data)
                VALUES (?, ?, ?)
            """, (job_id, status.value, input_data))
            
            created = cursor.rowcount > 0
            
            if created:
                logger.debug(f"Registered job: {job_id}")
            
            return created
    
    def register_batch(
        self,
        items: List[Any],
        id_func: Optional[callable] = None
    ) -> int:
        """
        Register multiple jobs in batch
        
        Args:
            items: List of items to process
            id_func: Function to generate job_id from item (default: hash)
        
        Returns: Number of new jobs registered
        
        Production Pattern: Batch INSERT for performance
        """
        if id_func is None:
            id_func = self._generate_job_id
        
        with self._get_connection() as conn:
            # Prepare batch data
            jobs_data = [
                (id_func(item), JobStatus.PENDING.value, str(item))
                for item in items
            ]
            
            # Batch insert
            conn.executemany("""
                INSERT OR IGNORE INTO jobs (job_id, status, input_data)
                VALUES (?, ?, ?)
            """, jobs_data)
            
            created = conn.total_changes
            
            logger.info(f"Registered {created}/{len(items)} new jobs")
            
            return created
    
    # ========================================
    # JOB STATUS MANAGEMENT
    # ========================================
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of a job"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return JobStatus(row['status'])
            return None
    
    def mark_processing(self, job_id: str) -> None:
        """Mark job as currently processing"""
        self._update_status(job_id, JobStatus.PROCESSING)
    
    def mark_complete(
        self,
        job_id: str,
        result_data: Optional[str] = None
    ) -> None:
        """
        Mark job as successfully completed
        
        Args:
            job_id: Job identifier
            result_data: Optional result to store
        """
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE jobs 
                SET status = ?, 
                    result_data = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (JobStatus.COMPLETE.value, result_data, job_id))
            
            logger.debug(f"✓ Job complete: {job_id}")
    
    def mark_failed(
        self,
        job_id: str,
        error: str,
        increment_attempts: bool = True
    ) -> None:
        """
        Mark job as failed
        
        Args:
            job_id: Job identifier
            error: Error message
            increment_attempts: Whether to increment attempt counter
        """
        with self._get_connection() as conn:
            if increment_attempts:
                conn.execute("""
                    UPDATE jobs 
                    SET status = ?, 
                        error = ?,
                        attempts = attempts + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (JobStatus.FAILED.value, error, job_id))
            else:
                conn.execute("""
                    UPDATE jobs 
                    SET status = ?, 
                        error = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (JobStatus.FAILED.value, error, job_id))
            
            logger.warning(f"✗ Job failed: {job_id} - {error}")
    
    def _update_status(self, job_id: str, status: JobStatus) -> None:
        """Update job status"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE jobs 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (status.value, job_id))
    
    # ========================================
    # QUERY METHODS
    # ========================================
    
    def is_complete(self, job_id: str) -> bool:
        """Check if job is already complete"""
        status = self.get_job_status(job_id)
        return status == JobStatus.COMPLETE
    
    def get_pending_jobs(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get list of pending jobs
        
        Production Pattern: Resumable processing
        Why: Fetch only jobs that need processing
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM jobs WHERE status = ?"
            params = [JobStatus.PENDING.value]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_failed_jobs(self, min_attempts: int = 1) -> List[Dict]:
        """
        Get list of failed jobs
        
        Args:
            min_attempts: Minimum attempts to filter (default: 1)
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM jobs 
                WHERE status = ? AND attempts >= ?
                ORDER BY updated_at DESC
            """, (JobStatus.FAILED.value, min_attempts))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get job statistics
        
        Production Pattern: Observable system
        Why: Monitoring and alerting need metrics
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM jobs
                GROUP BY status
            """)
            
            stats = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Add total
            stats['total'] = sum(stats.values())
            
            # Calculate completion rate
            complete = stats.get(JobStatus.COMPLETE.value, 0)
            total = stats['total']
            stats['completion_rate'] = (complete / total * 100) if total > 0 else 0
            
            return stats
    
    def print_statistics(self) -> None:
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("JOB TRACKER STATISTICS")
        print("=" * 50)
        print(f"Total jobs:    {stats['total']}")
        print(f"  ✓ Complete:  {stats.get(JobStatus.COMPLETE.value, 0)}")
        print(f"  ⏳ Pending:  {stats.get(JobStatus.PENDING.value, 0)}")
        print(f"  🔄 Processing: {stats.get(JobStatus.PROCESSING.value, 0)}")
        print(f"  ✗ Failed:    {stats.get(JobStatus.FAILED.value, 0)}")
        print(f"\nCompletion rate: {stats['completion_rate']:.1f}%")
        print("=" * 50)
    
    # ========================================
    # CLEANUP METHODS
    # ========================================
    
    def reset_processing_jobs(self) -> int:
        """
        Reset jobs stuck in 'processing' state
        
        Production Pattern: Crash recovery
        Why: If process crashes, jobs stay in 'processing' forever
        Call this on startup to recover
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE jobs 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE status = ?
            """, (JobStatus.PENDING.value, JobStatus.PROCESSING.value))
            
            reset_count = cursor.rowcount
            
            if reset_count > 0:
                logger.info(f"Reset {reset_count} stuck processing jobs")
            
            return reset_count
    
    def retry_failed_jobs(self, max_attempts: int = 3) -> int:
        """
        Reset failed jobs with attempts < max_attempts
        
        Production Pattern: Automatic retry
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE jobs 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE status = ? AND attempts < ?
            """, (JobStatus.PENDING.value, JobStatus.FAILED.value, max_attempts))
            
            retry_count = cursor.rowcount
            
            if retry_count > 0:
                logger.info(f"Retrying {retry_count} failed jobs")
            
            return retry_count
    
    def clear_all_jobs(self) -> None:
        """Clear all jobs (use with caution!)"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM jobs")
            logger.warning("Cleared all jobs from database")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("JOB TRACKER DEMONSTRATIONS")
    print("=" * 60)
    
    # Initialize tracker
    tracker = JobTracker("demo_jobs.db")
    tracker.clear_all_jobs()  # Start fresh
    
    # Example 1: Register jobs
    print("\n1. REGISTERING JOBS")
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5"
    ]
    
    tracker.register_batch(urls)
    tracker.print_statistics()
    
    # Example 2: Process jobs with state tracking
    print("\n2. PROCESSING JOBS")
    
    def process_url(url: str) -> str:
        """Simulate processing"""
        time.sleep(0.2)
        if "page3" in url:
            raise ValueError("Simulated error on page3")
        return f"Processed {url}"
    
    for url in urls:
        job_id = tracker._generate_job_id(url)
        
        # Skip if already complete (idempotent!)
        if tracker.is_complete(job_id):
            print(f"⏭️  Skipping {url} (already complete)")
            continue
        
        tracker.mark_processing(job_id)
        
        try:
            result = process_url(url)
            tracker.mark_complete(job_id, result)
            print(f"✓ Completed {url}")
        except Exception as e:
            tracker.mark_failed(job_id, str(e))
            print(f"✗ Failed {url}: {e}")
    
    tracker.print_statistics()
    
    # Example 3: Resume capability
    print("\n3. RESUME CAPABILITY (Re-running same jobs)")
    
    # Re-run the loop - completed jobs will be skipped!
    for url in urls:
        job_id = tracker._generate_job_id(url)
        
        if tracker.is_complete(job_id):
            print(f"⏭️  Skipping {url} (already complete)")
        else:
            print(f"🔄 Would process {url}")
    
    # Example 4: Query failed jobs
    print("\n4. QUERYING FAILED JOBS")
    failed = tracker.get_failed_jobs()
    print(f"Found {len(failed)} failed jobs:")
    for job in failed:
        print(f"  - {job['input_data']}: {job['error']}")
    
    # Example 5: Retry failed jobs
    print("\n5. RETRYING FAILED JOBS")
    tracker.retry_failed_jobs(max_attempts=3)
    tracker.print_statistics()
    
    print("\n✅ Demo complete! Database saved to: demo_jobs.db")