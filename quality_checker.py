"""
Production-Grade Data Quality Checker
Week 2, Day 2, Session 2 - AI Generalist Training

Quality Gates:
1. Duplicate Detection: Exact + Near-duplicate
2. Completeness: Required fields, non-empty values
3. Drift Detection: Statistical anomalies
"""

import hashlib
import json
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from schemas import ScrapedDocument, ValidationError, ValidationReport, ValidationStatus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class QualityMetrics:
    """
    Track quality metrics over time
    
    Production Pattern: Store baseline stats for drift detection
    """
    avg_word_count: float = 0.0
    avg_content_length: float = 0.0
    total_documents: int = 0
    unique_sources: Set[str] = field(default_factory=set)
    
    # For drift detection
    word_count_history: List[float] = field(default_factory=list)
    content_length_history: List[float] = field(default_factory=list)


@dataclass
class DuplicateResult:
    """Result of duplicate check"""
    is_duplicate: bool
    duplicate_type: Optional[str] = None  # "exact" or "near"
    similarity_score: Optional[float] = None
    existing_id: Optional[str] = None


# ============================================================
# CORE QUALITY CHECKER
# ============================================================

class DataQualityChecker:
    """
    Production-grade data quality validation
    
    Architecture:
    - Stateful: Tracks seen documents, metrics over time
    - Layered: Each check is independent, composable
    - Observable: Detailed logging + metrics
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.30,  # 30% deviation triggers alert
        similarity_threshold: float = 0.95  # 95% similar = near-duplicate
    ):
        # Configuration
        self.drift_threshold = drift_threshold
        self.similarity_threshold = similarity_threshold
        
        # State tracking
        self.seen_content_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.metrics = QualityMetrics()
        
        logger.info("DataQualityChecker initialized")
        logger.info(f"  Drift threshold: {drift_threshold:.0%}")
        logger.info(f"  Similarity threshold: {similarity_threshold:.0%}")
    
    # ========================================
    # DUPLICATE DETECTION
    # ========================================
    
    def _compute_content_hash(self, content: str) -> str:
        """
        Compute hash of normalized content
        
        Production Pattern: Normalize before hashing to catch
        duplicates that differ only in whitespace/case
        """
        # Normalize: lowercase, remove extra whitespace
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def check_exact_duplicate(self, doc: ScrapedDocument) -> DuplicateResult:
        """
        Check for exact content duplicates
        
        Why Fast: O(1) hash lookup, handles millions of docs
        """
        content_hash = self._compute_content_hash(doc.content)
        
        if content_hash in self.seen_content_hashes:
            logger.warning(f"Exact duplicate detected: {doc.url}")
            return DuplicateResult(
                is_duplicate=True,
                duplicate_type="exact",
                similarity_score=1.0
            )
        
        # Not a duplicate - track it
        self.seen_content_hashes.add(content_hash)
        return DuplicateResult(is_duplicate=False)
    
    def check_url_duplicate(self, doc: ScrapedDocument) -> bool:
        """
        Check if URL was already processed
        
        Production Pattern: Separate from content hash because:
        - Same URL can change content over time (want updates)
        - But usually indicates re-scraping same page
        """
        url_str = str(doc.url)
        if url_str in self.seen_urls:
            logger.info(f"URL already processed: {url_str}")
            return True
        
        self.seen_urls.add(url_str)
        return False
    
    def check_near_duplicate(
        self,
        doc: ScrapedDocument,
        embedding: Optional[np.ndarray] = None
    ) -> DuplicateResult:
        """
        Check for near-duplicates using embeddings
        
        Production Trade-off:
        - More expensive than hash check (embedding comparison)
        - Catches paraphrased/slightly modified content
        - Use only when exact duplicate check passes
        
        Args:
            embedding: Pre-computed embedding (pass if available to save compute)
        """
        if embedding is None:
            # In production, you'd call your embedding model here
            # For now, we'll simulate with content length vector
            embedding = self._mock_embedding(doc.content)
        
        doc_id = doc.doc_id or self._compute_content_hash(doc.content)
        
        # Compare with all existing embeddings
        for existing_id, existing_emb in self.document_embeddings.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                existing_emb.reshape(1, -1)
            )[0][0]
            
            if similarity >= self.similarity_threshold:
                logger.warning(
                    f"Near-duplicate detected: {doc.url} "
                    f"(similarity: {similarity:.2%} with {existing_id})"
                )
                return DuplicateResult(
                    is_duplicate=True,
                    duplicate_type="near",
                    similarity_score=similarity,
                    existing_id=existing_id
                )
        
        # Not a duplicate - store for future comparisons
        self.document_embeddings[doc_id] = embedding
        return DuplicateResult(is_duplicate=False)
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """
        Mock embedding for demo purposes
        
        Production: Replace with actual embedding model
        (Gemini, OpenAI, sentence-transformers)
        """
        # Simple feature vector based on text statistics
        features = [
            len(text),
            len(text.split()),
            text.count('.'),
            text.count(','),
            sum(c.isupper() for c in text)
        ]
        return np.array(features, dtype=np.float32)
    
    # ========================================
    # COMPLETENESS CHECKS
    # ========================================
    
    def check_completeness(self, doc: ScrapedDocument) -> Tuple[bool, List[str]]:
        """
        Verify all required fields are present and meaningful
        
        Production Pattern: Return (is_complete, issues) not just bool
        Why: Helps with debugging and targeted fixes
        """
        issues = []
        
        # Check required string fields aren't empty
        if not doc.title or not doc.title.strip():
            issues.append("title is empty")
        
        if not doc.content or not doc.content.strip():
            issues.append("content is empty")
        
        if not str(doc.url):
            issues.append("url is missing")
        
        # Check word count matches content
        actual_word_count = len(doc.content.split())
        reported_word_count = doc.word_count
        
        # Allow 10% tolerance for word count discrepancies
        if abs(actual_word_count - reported_word_count) > actual_word_count * 0.1:
            issues.append(
                f"word_count mismatch: reported {reported_word_count}, "
                f"actual {actual_word_count}"
            )
        
        # Check metadata completeness
        if not doc.metadata:
            issues.append("metadata is empty")
        elif 'char_count' not in doc.metadata:
            issues.append("metadata missing char_count")
        
        is_complete = len(issues) == 0
        
        if not is_complete:
            logger.warning(f"Completeness check failed for {doc.url}: {issues}")
        
        return is_complete, issues
    
    # ========================================
    # DRIFT DETECTION
    # ========================================
    
    def update_baseline_metrics(self, doc: ScrapedDocument) -> None:
        """
        Update baseline metrics with new document
        
        Production Pattern: Running statistics, not batch recalculation
        Why: O(1) update vs O(n) recalculation
        """
        # Update running averages
        n = self.metrics.total_documents
        
        self.metrics.avg_word_count = (
            (self.metrics.avg_word_count * n + doc.word_count) / (n + 1)
        )
        
        content_length = len(doc.content)
        self.metrics.avg_content_length = (
            (self.metrics.avg_content_length * n + content_length) / (n + 1)
        )
        
        self.metrics.total_documents += 1
        
        # FIX: doc.source is already a string (thanks to use_enum_values=True in schema)
        self.metrics.unique_sources.add(doc.source)
        
        # Track history for drift detection
        self.metrics.word_count_history.append(doc.word_count)
        self.metrics.content_length_history.append(content_length)
    
    def check_drift(self, doc: ScrapedDocument) -> Tuple[bool, Dict[str, float]]:
        """
        Detect if document deviates significantly from baseline
        
        Production Pattern: Z-score for drift detection
        Why: Handles varying scales, statistical significance
        
        Returns: (has_drift, drift_details)
        """
        if self.metrics.total_documents < 10:
            # Need baseline - not enough data yet
            return False, {}
        
        drift_details = {}
        has_drift = False
        
        # Check word count drift
        word_count_deviation = abs(
            doc.word_count - self.metrics.avg_word_count
        ) / self.metrics.avg_word_count
        
        drift_details['word_count_deviation'] = word_count_deviation
        
        if word_count_deviation > self.drift_threshold:
            has_drift = True
            logger.warning(
                f"Word count drift detected: {doc.word_count} vs "
                f"avg {self.metrics.avg_word_count:.0f} "
                f"({word_count_deviation:.1%} deviation)"
            )
        
        # Check content length drift
        content_length = len(doc.content)
        length_deviation = abs(
            content_length - self.metrics.avg_content_length
        ) / self.metrics.avg_content_length
        
        drift_details['length_deviation'] = length_deviation
        
        if length_deviation > self.drift_threshold:
            has_drift = True
            logger.warning(
                f"Content length drift detected: {content_length} vs "
                f"avg {self.metrics.avg_content_length:.0f} "
                f"({length_deviation:.1%} deviation)"
            )
        
        return has_drift, drift_details
    
    # ========================================
    # FULL QUALITY CHECK
    # ========================================
    
    def validate_document(
        self,
        doc: ScrapedDocument,
        check_duplicates: bool = True,
        check_drift: bool = True
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Run all quality checks on a document
        
        Production Pattern: Fail fast, collect all errors
        Why: Don't waste time on bad data, but report everything wrong
        
        Returns: (is_valid, errors)
        """
        errors = []
        
        # 1. Completeness (always check)
        is_complete, issues = self.check_completeness(doc)
        if not is_complete:
            for issue in issues:
                errors.append(ValidationError(
                    record_id=doc.doc_id,
                    field=issue.split()[0],
                    error_type="completeness",
                    message=issue,
                    severity="high"
                ))
        
        # 2. Duplicates (if enabled)
        if check_duplicates:
            # Exact duplicate
            exact_dup = self.check_exact_duplicate(doc)
            if exact_dup.is_duplicate:
                errors.append(ValidationError(
                    record_id=doc.doc_id,
                    error_type="duplicate",
                    message=f"Exact duplicate of existing content",
                    severity="medium"
                ))
            
            # URL duplicate
            if self.check_url_duplicate(doc):
                errors.append(ValidationError(
                    record_id=doc.doc_id,
                    field="url",
                    error_type="duplicate",
                    message=f"URL already processed: {doc.url}",
                    severity="low"  # Lower severity - might be intentional update
                ))
        
        # 3. Drift detection (if enabled and not a duplicate)
        if check_drift and not errors:  # Only check if passed other tests
            has_drift, drift_details = self.check_drift(doc)
            if has_drift:
                errors.append(ValidationError(
                    record_id=doc.doc_id,
                    error_type="anomaly",
                    message=f"Data drift detected: {drift_details}",
                    severity="medium"
                ))
        
        # Update metrics if document is valid
        if not errors:
            self.update_baseline_metrics(doc)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    # ========================================
    # BATCH PROCESSING
    # ========================================
    
    def validate_batch(
        self,
        documents: List[ScrapedDocument]
    ) -> ValidationReport:
        """
        Validate a batch of documents
        
        Production Pattern: Batch processing with reporting
        Why: Process thousands of docs, get actionable summary
        """
        report = ValidationReport(
            total_processed=len(documents),
            passed=0,
            failed=0,
            status=ValidationStatus.PASSED
        )
        
        logger.info(f"Validating batch of {len(documents)} documents...")
        
        for doc in documents:
            is_valid, errors = self.validate_document(doc)
            
            if is_valid:
                report.passed += 1
            else:
                report.failed += 1
                report.errors.extend(errors)
        
        # Determine overall status
        if report.failed == 0:
            report.status = ValidationStatus.PASSED
        elif report.pass_rate >= 90:
            report.status = ValidationStatus.WARNING
        else:
            report.status = ValidationStatus.FAILED
        
        report.completed_at = datetime.now()
        
        logger.info(f"Batch validation complete:")
        logger.info(f"  Passed: {report.passed}/{report.total_processed}")
        logger.info(f"  Failed: {report.failed}")
        logger.info(f"  Pass rate: {report.pass_rate:.1f}%")
        
        return report


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    from datetime import timedelta
    
    print("=" * 60)
    print("DATA QUALITY CHECKER DEMO")
    print("=" * 60)
    
    checker = DataQualityChecker()
    
    # Create test documents
    base_doc_data = {
        "url": "https://example.com/article1",
        "title": "AI in Healthcare",
        "content": "Artificial intelligence is transforming healthcare. " * 30,
        "scraped_at": datetime.now(),
        "source": "web_scrape",
        "word_count": 150,
        "doc_id": "doc_001",
        "metadata": {"char_count": 1500}
    }
    
    print("\n1. VALID DOCUMENT")
    doc1 = ScrapedDocument(**base_doc_data)
    is_valid, errors = checker.validate_document(doc1)
    print(f"Valid: {is_valid}, Errors: {len(errors)}")
    
    print("\n2. EXACT DUPLICATE")
    doc2 = ScrapedDocument(**{**base_doc_data, "doc_id": "doc_002"})
    is_valid, errors = checker.validate_document(doc2)
    print(f"Valid: {is_valid}, Errors: {len(errors)}")
    if errors:
        print(f"  Error: {errors[0].message}")
    
    print("\n3. INCOMPLETE DOCUMENT")
    incomplete_data = {**base_doc_data, "doc_id": "doc_003", "title": "", "url": "https://example.com/article3"}
    doc3 = ScrapedDocument(**incomplete_data)
    is_valid, errors = checker.validate_document(doc3)
    print(f"Valid: {is_valid}, Errors: {len(errors)}")
    for error in errors:
        print(f"  - {error.message}")
    
    print("\n4. DRIFT DETECTION")
    # Add baseline docs
    for i in range(10):
        baseline_data = {
            **base_doc_data,
            "doc_id": f"baseline_{i}",
            "url": f"https://example.com/baseline{i}",
            "content": "Normal content. " * 25,
            "word_count": 50
        }
        baseline_doc = ScrapedDocument(**baseline_data)
        checker.validate_document(baseline_doc)
    
    # Now add anomalous doc
    anomaly_data = {
        **base_doc_data,
        "doc_id": "anomaly_001",
        "url": "https://example.com/anomaly",
        "content": "Very short.",
        "word_count": 2
    }
    anomaly_doc = ScrapedDocument(**anomaly_data)
    is_valid, errors = checker.validate_document(anomaly_doc, check_drift=True)
    print(f"Valid: {is_valid}, Errors: {len(errors)}")
    if errors:
        print(f"  Error: {errors[0].message}")
    
    print("\n5. BATCH VALIDATION")
    batch = [doc1, doc2, doc3]
    report = checker.validate_batch(batch)
    print(report.summary())