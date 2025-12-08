"""
FINAL DELIVERABLE: Production-Validated Pipeline
Week 2, Day 2, Session 4 - AI Generalist Training

This script integrates:
1. Schema Validation (Pydantic)
2. Quality Checks (Deduplication + Drift)
3. Anomaly Detection (Spam + PII)
4. Alerting (Rate-limited notifications)
"""

import time
import logging
from datetime import datetime
from typing import List, Dict

# Import our 3 custom modules
from schemas import ScrapedDocument, ValidationError, ValidationReport, ValidationStatus
from quality_checker import DataQualityChecker
from alerting import AlertManager, TextAnomalyDetector, AlertConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")

class ProductionPipeline:
    def __init__(self):
        # 1. Initialize all our defenses
        self.quality_checker = DataQualityChecker()
        self.anomaly_detector = TextAnomalyDetector()
        self.alerter = AlertManager(config=AlertConfig(min_severity="medium"))
        
        logger.info("🛡️  Production Pipeline Initialized")

    def process_document(self, raw_data: Dict) -> bool:
        """
        Runs a single document through the Gauntlet.
        Returns True if it survived, False if rejected.
        """
        doc_id = raw_data.get("doc_id", "unknown")
        logger.info(f"Processing {doc_id}...")

        # ----------------------------------------
        # STAGE 1: Schema Validation (The Bouncer)
        # ----------------------------------------
        try:
            # This line will auto-validate types, lengths, and URLs
            doc = ScrapedDocument(**raw_data)
        except Exception as e:
            # If Pydantic fails, we create a Schema Error
            error = ValidationError(
                record_id=doc_id,
                field="schema",
                error_type="schema_violation",
                message=str(e),
                severity="high"
            )
            self.alerter.send_alert(error)
            return False

        # ----------------------------------------
        # STAGE 2: Quality Gates (The Investigator)
        # ----------------------------------------
        # Check for duplicates, empty content, and statistical drift
        is_valid, quality_errors = self.quality_checker.validate_document(doc)
        
        if not is_valid:
            for err in quality_errors:
                self.alerter.send_alert(err)
            return False # Stop processing

        # ----------------------------------------
        # STAGE 3: Anomaly Detection (The Security Scanner)
        # ----------------------------------------
        # Check for spam, PII, and screaming text
        anomalies = self.anomaly_detector.scan(doc)
        if anomalies:
            for anomaly in anomalies:
                self.alerter.send_alert(anomaly)
            # We don't necessarily stop for anomalies, but we alert.
            # (Decide based on your business logic. Here, we'll continue with a warning)

        # ----------------------------------------
        # STAGE 4: Success!
        # ----------------------------------------
        logger.info(f"✅ Document {doc.doc_id} accepted into Knowledge Base.")
        return True

# ============================================================
# RUN THE SIMULATION
# ============================================================
if __name__ == "__main__":
    pipeline = ProductionPipeline()
    
    # We add 'scraped_at' to satisfy the Strict Schema
    current_time = datetime.now()

    test_batch = [
        # 1. PERFECT DOCUMENT (Updated with metadata)
        {
            "doc_id": "doc_perfect",
            "title": "The Future of AI",
            "url": "https://example.com/ai",
            "content": "Artificial Intelligence is evolving rapidly. " * 10,
            "word_count": 50,
            "source": "web_scrape",
            "scraped_at": current_time,
            "metadata": {"char_count": 500, "author": "Neeraj"} 
        },
        # 2. SCHEMA FAIL (Bad URL + No Word Count)
        {
            "doc_id": "doc_bad_schema",
            "title": "Bad URL Doc",
            "url": "not-a-url", 
            "content": "Some content here.",
            "source": "web_scrape",
            "scraped_at": current_time,
            "metadata": {"char_count": 20}
        },
        # 3. QUALITY FAIL (Duplicate of doc_perfect)
        {
            "doc_id": "doc_duplicate",
            "title": "The Future of AI",
            "url": "https://example.com/ai",
            "content": "Artificial Intelligence is evolving rapidly. " * 10, 
            "word_count": 50,     # This lets it pass Schema so Quality Checker can catch it
            "source": "web_scrape",
            "scraped_at": current_time,
            "metadata": {"char_count": 500}
        },
        # 4. ANOMALY FAIL (Spam + Screaming)
        {
            "doc_id": "doc_spam",
            "title": "FREE MONEY",
            "url": "https://example.com/spam",
            "content": "CLICK HERE FOR VIAGRA AND FREE MONEY " * 20, 
            "word_count": 140,
            "source": "web_scrape",
            "scraped_at": current_time,
            "metadata": {"char_count": 1000}
        }
    ]

    print("\n" + "="*50)
    print("🚀 STARTING PIPELINE BATTLE TEST")
    print("="*50 + "\n")

    accepted_count = 0
    for data in test_batch:
        if pipeline.process_document(data):
            accepted_count += 1
        time.sleep(0.5) # Formatting pause

    print("\n" + "="*50)
    print(f"🏁 RESULTS: Accepted {accepted_count}/{len(test_batch)} documents.")
    print("="*50)