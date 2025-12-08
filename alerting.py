"""
Production-Grade Alerting & Anomaly Detection
Week 2, Day 2, Session 3 - AI Generalist Training

Features:
- Rate Limiting: Don't spam Slack with 1000 identical errors
- De-duplication: Group similar alerts
- Multi-Channel: Console, File, Mock-Slack
- Text Analysis: Catch "screaming" text or spam patterns
"""

import logging
import re
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

from schemas import ScrapedDocument, ValidationError

# Configure logging
logging.basicConfig(
    filename='pipeline_alerts.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# TEXT ANOMALY DETECTOR
# ============================================================

class TextAnomalyDetector:
    """
    Scans text for suspicious patterns that might not be 'errors'
    but indicate low quality (spam, screaming, PII).
    """
    
    def __init__(self):
        # Compiled regex for performance
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.spam_keywords = {'buy now', 'click here', 'viagra', 'casino', 'free money'}

    def scan(self, doc: ScrapedDocument) -> List[ValidationError]:
        anomalies = []
        text = doc.content
        
        # 1. Check for "Screaming" (All Caps)
        # We check if >60% of letters are uppercase (ignoring numbers/spaces)
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 50:  # Only check significant text
            upper_count = sum(1 for c in letters if c.isupper())
            ratio = upper_count / len(letters)
            
            if ratio > 0.6:
                anomalies.append(ValidationError(
                    record_id=doc.doc_id,
                    field="content",
                    error_type="anomaly_text_quality",
                    message=f"Text is screaming ({ratio:.1%} uppercase)",
                    severity="medium"
                ))

        # 2. Check for PII (Personally Identifiable Information)
        # In production, you might scrub this, here we just flag it.
        if self.email_pattern.search(text):
             anomalies.append(ValidationError(
                    record_id=doc.doc_id,
                    field="content",
                    error_type="anomaly_pii",
                    message="Potential PII detected: Email address found",
                    severity="high"
                ))

        # 3. Check for Spam Keywords
        found_keywords = [kw for kw in self.spam_keywords if kw in text.lower()]
        if found_keywords:
            anomalies.append(ValidationError(
                record_id=doc.doc_id,
                field="content",
                error_type="anomaly_spam",
                message=f"Spam keywords detected: {found_keywords}",
                severity="high"
            ))

        return anomalies

# ============================================================
# ALERT MANAGER
# ============================================================

@dataclass
class AlertConfig:
    slack_webhook_url: Optional[str] = None
    min_severity: str = "medium"  # low, medium, high, critical
    rate_limit_seconds: int = 300  # 5 minutes

class AlertManager:
    """
    Central hub for dispatching notifications.
    Implements rate limiting so you don't wake up to 5,000 messages.
    """
    
    def __init__(self, config: AlertConfig = AlertConfig()):
        self.config = config
        self.last_alert_time: Dict[str, datetime] = defaultdict(lambda: datetime.min)
        self.alert_counts: Dict[str, int] = defaultdict(int)
        
        # Severity hierarchy for filtering
        self.severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def _should_send(self, error: ValidationError) -> bool:
        """Check rules to decide if we should trigger an external alert"""
        
        # 1. Severity Check
        error_level = self.severity_levels.get(error.severity, 1)
        config_level = self.severity_levels.get(self.config.min_severity, 1)
        if error_level < config_level:
            return False

        # 2. Rate Limit Check (Key = error_type + part of message)
        # We group similar errors so we don't alert 50 times for "Title missing"
        alert_key = f"{error.error_type}:{error.message[:20]}"
        now = datetime.now()
        
        time_since_last = (now - self.last_alert_time[alert_key]).total_seconds()
        
        if time_since_last < self.config.rate_limit_seconds:
            self.alert_counts[alert_key] += 1
            return False  # Suppress alert
            
        # Update trackers
        self.last_alert_time[alert_key] = now
        count = self.alert_counts[alert_key]
        self.alert_counts[alert_key] = 0  # Reset count
        
        if count > 0:
            logger.info(f"Suppressed {count} similar alerts for {alert_key}")
            
        return True

    def send_alert(self, error: ValidationError):
        """Dispatch the alert to configured channels"""
        
        # Always log to file (Disk is cheap)
        logger.warning(f"[{error.severity.upper()}] {error.error_type}: {error.message} (ID: {error.record_id})")
        
        # Check filters for external notifications
        if not self._should_send(error):
            return

        # Mock Slack/Email Notification
        self._send_to_slack(error)

    def _send_to_slack(self, error: ValidationError):
        """
        Mock function to simulate sending a Slack webhook.
        In a real job, you'd use 'requests.post(url, json=...)'
        """
        # Visual separator for the console to make it look like a notification
        print(f"\n🚨  [SLACK ALERT]  🚨")
        print(f"Severity: {error.severity.upper()}")
        print(f"Type:     {error.error_type}")
        print(f"Message:  {error.message}")
        print(f"Doc ID:   {error.record_id}")
        print(f"Time:     {error.timestamp.strftime('%H:%M:%S')}")
        print("-" * 30 + "\n")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("ALERTING SYSTEM DEMO")
    print("="*60)

    # 1. Setup
    detector = TextAnomalyDetector()
    alerter = AlertManager(config=AlertConfig(rate_limit_seconds=2)) # Short limit for demo

    # 2. Test Anomaly Detection
    print("\n--- Testing Text Analysis ---")
    
    # "Screaming" Document
    bad_doc = ScrapedDocument(
        title="YELLING",
        url="http://test.com",
        content="THIS IS URGENT YOU MUST CLICK HERE TO WIN A PRIZE " * 10,
        word_count=100,
        doc_id="doc_screaming"
    )
    
    anomalies = detector.scan(bad_doc)
    for a in anomalies:
        print(f"Caught: {a.message}")
        alerter.send_alert(a)

    # 3. Test Rate Limiting
    print("\n--- Testing Rate Limiting (Spamming 5 critical errors) ---")
    
    critical_error = ValidationError(
        error_type="database_connection", 
        message="Connection refused", 
        severity="critical"
    )

    for i in range(5):
        print(f"Sending error #{i+1}...")
        alerter.send_alert(critical_error)
        time.sleep(0.5)

    print("\nWaiting 3 seconds for rate limit to expire...")
    time.sleep(3)
    
    print("Sending error #6 (Should trigger now)...")
    alerter.send_alert(critical_error)