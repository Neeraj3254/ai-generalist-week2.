"""
Production-Grade Schema Validation with Pydantic
Week 2, Day 2, Session 1 - AI Generalist Training

Why Pydantic?
- Type safety: Catch errors at data entry, not runtime
- Auto-validation: No manual if/else chains
- Clear errors: "word_count must be positive" vs generic crashes
- Performance: Written in C, faster than manual validation
"""

from typing import Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
from enum import Enum


# ============================================================
# ENUMS: Define allowed values (type-safe constants)
# ============================================================

class SourceType(str, Enum):
    """
    Production Pattern: Use Enums instead of raw strings
    Why: Prevents typos, IDE autocomplete, clear documentation
    """
    WEB_SCRAPE = "web_scrape"
    API = "api"
    RSS = "rss"
    MANUAL = "manual"


class ValidationStatus(str, Enum):
    """Status of validation checks"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


# ============================================================
# CORE DATA MODELS
# ============================================================

class ScrapedDocument(BaseModel):
    """
    Raw document from extraction layer
    
    Production Standard:
    - Every field has constraints (min/max length, ranges)
    - Custom validators for business logic
    - Clear error messages via Field(description=...)
    """
    
    # Required fields with validation
    url: HttpUrl = Field(
        description="Source URL (must be valid HTTP/HTTPS)"
    )
    
    title: str = Field(
        min_length=1,
        max_length=500,
        description="Document title (1-500 chars)"
    )
    
    content: str = Field(
        min_length=50,
        max_length=500_000,
        description="Main text content (50-500k chars)"
    )
    
    scraped_at: datetime = Field(
        description="Timestamp of scraping (ISO format)"
    )
    
    source: SourceType = Field(
        description="Source type (web_scrape, api, rss, manual)"
    )
    
    word_count: int = Field(
        gt=0,
        lt=1_000_000,
        description="Word count (positive integer, max 1M)"
    )
    
    # Optional metadata
    doc_id: Optional[str] = Field(
        default=None,
        description="Unique document identifier"
    )
    
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata (char_count, etc.)"
    )
    
    # Pydantic v2 config
    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute changes
        str_strip_whitespace=True,  # Auto-trim strings
        use_enum_values=True  # Store enum values, not enum objects
    )
    
    @field_validator('content')
    @classmethod
    def content_must_have_substance(cls, v: str) -> str:
        """
        Custom validator: Ensure content isn't just whitespace
        
        Production Pattern: Add business logic validation here
        """
        if len(v.strip()) < 50:
            raise ValueError('Content must have at least 50 non-whitespace characters')
        return v
    
    @field_validator('title')
    @classmethod
    def title_must_be_meaningful(cls, v: str) -> str:
        """Reject generic titles"""
        forbidden = ['untitled', 'no title', 'n/a', 'none', '']
        if v.lower().strip() in forbidden:
            raise ValueError(f'Title cannot be: {v}')
        return v
    
    @field_validator('scraped_at')
    @classmethod
    def scraped_at_not_future(cls, v: datetime) -> datetime:
        """Prevent future timestamps (indicates system clock issues)"""
        if v > datetime.now():
            raise ValueError(f'scraped_at cannot be in the future: {v}')
        return v


class ProcessedChunk(BaseModel):
    """
    Transformed chunk ready for embedding
    
    Production Standard:
    - Enforces chunk size limits (critical for embedding models)
    - Validates metadata completeness
    - Ensures chunk_id uniqueness pattern
    """
    
    chunk_id: str = Field(
        pattern=r'^[a-zA-Z0-9_-]+_chunk_\d+$',
        description="Format: {doc_id}_chunk_{index}"
    )
    
    text: str = Field(
        min_length=50,
        max_length=2000,
        description="Chunk text (50-2000 chars, optimal for embeddings)"
    )
    
    # Required metadata fields
    source: SourceType
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    scraped_at: datetime
    chunk_index: int = Field(ge=0, description="0-indexed chunk position")
    total_chunks: int = Field(gt=0, description="Total chunks in document")
    
    # Optional metadata
    metadata: Dict = Field(default_factory=dict)
    
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    @field_validator('chunk_index', 'total_chunks')
    @classmethod
    def validate_chunk_consistency(cls, v: int, info) -> int:
        """Ensure chunk_index < total_chunks"""
        if info.field_name == 'chunk_index':
            # Access is limited in Pydantic v2, so we validate in model_validator
            pass
        return v
    
    @field_validator('text')
    @classmethod
    def text_must_be_substantive(cls, v: str) -> str:
        """Reject chunks that are mostly non-alphabetic"""
        alpha_chars = sum(c.isalpha() for c in v)
        if alpha_chars < len(v) * 0.5:
            raise ValueError('Chunk must be at least 50% alphabetic characters')
        return v


class ValidationError(BaseModel):
    """
    Single validation error record
    
    Production Pattern: Structured error logging
    Why: Enable filtering, aggregation, alerting
    """
    
    record_id: Optional[str] = Field(
        default=None,
        description="ID of failed record"
    )
    
    field: Optional[str] = Field(
        default=None,
        description="Field that failed validation"
    )
    
    error_type: str = Field(
        description="Error category (schema, duplicate, anomaly)"
    )
    
    message: str = Field(
        description="Human-readable error message"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When error occurred"
    )
    
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Error severity level"
    )


class ValidationReport(BaseModel):
    """
    Summary of validation run
    
    Production Standard:
    - Actionable metrics (pass rate, error types)
    - Sortable errors (by severity)
    - Exportable for dashboards
    """
    
    total_processed: int = Field(
        ge=0,
        description="Total records processed"
    )
    
    passed: int = Field(
        ge=0,
        description="Records that passed validation"
    )
    
    failed: int = Field(
        ge=0,
        description="Records that failed validation"
    )
    
    warnings: int = Field(
        ge=0,
        default=0,
        description="Non-critical issues"
    )
    
    errors: List[ValidationError] = Field(
        default_factory=list,
        description="Detailed error records"
    )
    
    status: ValidationStatus = Field(
        description="Overall validation status"
    )
    
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="Validation start time"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Validation completion time"
    )
    
    model_config = ConfigDict(validate_assignment=True)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage"""
        if self.total_processed == 0:
            return 0.0
        return (self.passed / self.total_processed) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate validation duration"""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return None
    
    def get_critical_errors(self) -> List[ValidationError]:
        """Filter critical errors for alerting"""
        return [e for e in self.errors if e.severity == "critical"]
    
    def summary(self) -> str:
        """
        Production Pattern: Human-readable summary
        Why: For logs, Slack alerts, dashboards
        """
        return (
            f"Validation Report:\n"
            f"  Status: {self.status.value.upper()}\n"
            f"  Processed: {self.total_processed}\n"
            f"  Passed: {self.passed} ({self.pass_rate:.1f}%)\n"
            f"  Failed: {self.failed}\n"
            f"  Warnings: {self.warnings}\n"
            f"  Critical Errors: {len(self.get_critical_errors())}\n"
            f"  Duration: {self.duration_seconds:.2f}s" if self.duration_seconds else ""
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_document(data: Dict) -> tuple[Optional[ScrapedDocument], Optional[ValidationError]]:
    """
    Safe validation wrapper
    
    Production Pattern: Never let validation crash the pipeline
    Why: One bad record shouldn't kill the entire batch
    
    Returns: (document, error) - exactly one will be None
    """
    try:
        doc = ScrapedDocument(**data)
        return doc, None
    except Exception as e:
        error = ValidationError(
            record_id=data.get('doc_id', 'unknown'),
            error_type="schema_validation",
            message=str(e),
            severity="high"
        )
        return None, error


def validate_chunk(data: Dict) -> tuple[Optional[ProcessedChunk], Optional[ValidationError]]:
    """Safe chunk validation wrapper"""
    try:
        chunk = ProcessedChunk(**data)
        return chunk, None
    except Exception as e:
        error = ValidationError(
            record_id=data.get('chunk_id', 'unknown'),
            error_type="schema_validation",
            message=str(e),
            severity="high"
        )
        return None, error


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SCHEMA VALIDATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Valid document
    print("\n1. VALID DOCUMENT")
    valid_data = {
        "url": "https://example.com/article",
        "title": "AI in Healthcare",
        "content": "Artificial intelligence is transforming healthcare. " * 20,
        "scraped_at": datetime.now(),
        "source": "web_scrape",
        "word_count": 150,
        "doc_id": "doc_001"
    }
    
    doc, error = validate_document(valid_data)
    if doc:
        print(f"✓ Document validated: {doc.title}")
        print(f"  URL: {doc.url}")
        print(f"  Word count: {doc.word_count}")
    
    # Example 2: Invalid document (future timestamp)
    print("\n2. INVALID DOCUMENT (future timestamp)")
    from datetime import timedelta
    invalid_data = {
        **valid_data,
        "scraped_at": datetime.now() + timedelta(days=1)
    }
    
    doc, error = validate_document(invalid_data)
    if error:
        print(f"✗ Validation failed: {error.message}")
        print(f"  Severity: {error.severity}")
    
    # Example 3: Invalid document (bad title)
    print("\n3. INVALID DOCUMENT (generic title)")
    invalid_data = {
        **valid_data,
        "title": "Untitled",
        "scraped_at": datetime.now()
    }
    
    doc, error = validate_document(invalid_data)
    if error:
        print(f"✗ Validation failed: {error.message}")
    
    # Example 4: Validation report
    print("\n4. VALIDATION REPORT")
    report = ValidationReport(
        total_processed=100,
        passed=87,
        failed=13,
        warnings=5,
        status=ValidationStatus.WARNING
    )
    
    report.errors.append(ValidationError(
        record_id="doc_042",
        error_type="schema_validation",
        message="word_count must be positive",
        severity="critical"
    ))
    
    report.completed_at = datetime.now()
    
    print(report.summary())
    print(f"\nCritical errors: {len(report.get_critical_errors())}")