"""
Pydantic models for the workflow plan caching system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CachedPlan(BaseModel):
    """A cached workflow plan that can be reused for similar goals."""
    
    cache_id: str
    canonical_goal: str  # Normalized goal for matching
    goal_keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    goal_embedding: List[float] = Field(default_factory=list)  # Vector embedding for semantic search
    target_domain: Optional[str] = None  # e.g., "shopify", "instagram"
    target_url: Optional[str] = None
    page_context: Optional[str] = None  # "homepage", "login", "dashboard"
    planned_steps: List[Dict[str, Any]] = Field(default_factory=list)  # The cached PlannedStep objects
    total_steps: int = 0
    success_count: int = 0  # Times plan succeeded
    failure_count: int = 0  # Times plan failed
    total_uses: int = 0
    avg_completion_rate: float = 0.0  # 0.0-1.0
    user_corrections: List[Dict[str, Any]] = Field(default_factory=list)  # Accumulated corrections
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # For TTL
    original_session_id: str = ""
    original_user_goal: str = ""


class CacheMatchResult(BaseModel):
    """Result of a cache lookup operation."""
    
    hit: bool = False
    cache_id: Optional[str] = None
    confidence: float = 0.0  # 0.0-1.0
    match_method: str = "none"  # "exact", "keyword", "semantic", "none"
    cached_plan: Optional[CachedPlan] = None


class CacheStats(BaseModel):
    """Statistics about the cache."""
    
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    avg_completion_rate: float = 0.0
    entries_by_domain: Dict[str, int] = Field(default_factory=dict)
    entries_by_match_method: Dict[str, int] = Field(default_factory=dict)
