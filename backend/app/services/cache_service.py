"""
Caching service for workflow plans.

Stores successful step-by-step plans and retrieves them for similar future requests
to avoid redundant LLM calls.
"""
from __future__ import annotations

import logging
import math
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.cache import CachedPlan, CacheMatchResult, CacheStats
from app.services.embeddings import EmbeddingsError, embed_text

logger = logging.getLogger(__name__)

# Configuration constants
CACHE_TTL_DAYS = 30
MIN_SIMILARITY_THRESHOLD = 0.85  # For semantic matching
MIN_KEYWORD_MATCH_RATIO = 0.6    # For keyword matching
MIN_COMPLETION_RATE = 0.5        # Only use plans above this success rate

# Stopwords to remove when extracting keywords
_STOPWORDS = {
    "i", "want", "to", "the", "a", "an", "on", "in", "for", "and", "or",
    "please", "help", "me", "can", "you", "how", "do", "make", "create",
    "go", "get", "find", "search", "look", "need", "would", "like", "my",
    "this", "that", "with", "from", "into", "is", "are", "be", "been",
    "was", "were", "will", "have", "has", "had", "it", "its", "of",
}


def _utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def _extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from goal text.
    
    - Normalize text (lowercase, remove punctuation)
    - Remove stopwords
    - Keep words with length >= 3
    - Return top 10 unique keywords
    """
    if not text:
        return []
    
    # Normalize: lowercase and remove punctuation
    normalized = text.lower()
    normalized = normalized.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    
    # Split into words and filter
    words = normalized.split()
    keywords: List[str] = []
    seen: set = set()
    
    for word in words:
        word = word.strip()
        if (
            len(word) >= 3
            and word not in _STOPWORDS
            and word not in seen
            and not word.isdigit()
        ):
            keywords.append(word)
            seen.add(word)
    
    return keywords[:10]


def _calculate_keyword_similarity(keywords1: List[str], keywords2: List[str]) -> float:
    """
    Calculate Jaccard similarity: intersection / union.
    
    Returns 0.0 if either list is empty.
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns 0.0 if either vector is empty or has zero magnitude.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


async def ensure_cache_indexes(db) -> None:
    """
    Create MongoDB indexes for the plan_cache collection.
    
    Indexes:
    - TTL index on expires_at for auto-expiration
    - Text index on canonical_goal, goal_keywords, original_user_goal
    - Compound index on target_domain + goal_keywords
    - Index on canonical_goal for exact match
    - Compound index on target_domain + avg_completion_rate + total_uses for quality sorting
    """
    try:
        collection = db.plan_cache
        
        # TTL index for auto-expiration
        await collection.create_index(
            "expires_at",
            expireAfterSeconds=0,
            name="ttl_expires_at"
        )
        
        # Text index for full-text search
        await collection.create_index(
            [
                ("canonical_goal", "text"),
                ("original_user_goal", "text"),
                ("goal_keywords", "text"),
            ],
            name="text_search_idx"
        )
        
        # Exact match on canonical_goal
        await collection.create_index(
            "canonical_goal",
            name="canonical_goal_idx"
        )
        
        # Exact match on original_user_goal (raw prompt)
        await collection.create_index(
            "original_user_goal",
            name="original_user_goal_idx"
        )
        
        # Compound index for domain + quality sorting
        await collection.create_index(
            [
                ("target_domain", 1),
                ("avg_completion_rate", -1),
                ("total_uses", -1),
            ],
            name="domain_quality_idx"
        )
        
        # Index on cache_id for lookups
        await collection.create_index(
            "cache_id",
            unique=True,
            name="cache_id_idx"
        )
        
        logger.info("Cache indexes created successfully")
    except Exception as e:
        logger.warning(f"Failed to create cache indexes (non-fatal): {e}")


async def lookup_exact_match(
    db,
    canonical_goal: str,
    user_goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fast exact-match lookup on canonical_goal or original_user_goal.
    
    This is the fastest possible cache lookup - just indexed queries.
    No embedding generation or keyword extraction needed.
    
    Tries two strategies:
    1. Match on canonical_goal (normalized goal)
    2. Match on original_user_goal (exact raw prompt)
    
    Returns:
        Dict with keys: hit, cache_id, confidence, match_method, cached_plan
    """
    result: Dict[str, Any] = {
        "hit": False,
        "cache_id": None,
        "confidence": 0.0,
        "match_method": "none",
        "cached_plan": None,
    }
    
    try:
        collection = db.plan_cache
        
        # Strategy 1: Exact match on canonical_goal - uses index, very fast
        exact_match = await collection.find_one({
            "canonical_goal": canonical_goal,
            "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
        })
        
        if exact_match:
            logger.info(f"Cache EXACT HIT on canonical_goal (instant): {exact_match.get('cache_id')}")
            return {
                "hit": True,
                "cache_id": exact_match.get("cache_id"),
                "confidence": 1.0,
                "match_method": "exact",
                "cached_plan": exact_match,
            }
        
        # Strategy 2: Exact match on original_user_goal (raw prompt)
        # Normalize both the query and stored values for case-insensitive matching
        if user_goal:
            # Normalize: lowercase and collapse multiple spaces
            normalized_goal = " ".join(user_goal.lower().split())
            
            # Try exact match first (fastest)
            exact_match = await collection.find_one({
                "original_user_goal": user_goal,
                "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
            })
            
            if exact_match:
                logger.info(f"Cache EXACT HIT on raw prompt (instant): {exact_match.get('cache_id')}")
                return {
                    "hit": True,
                    "cache_id": exact_match.get("cache_id"),
                    "confidence": 1.0,
                    "match_method": "exact_raw",
                    "cached_plan": exact_match,
                }
            
            # If no exact match, try normalized match (case-insensitive, whitespace-agnostic)
            # Fetch candidates and check normalized versions
            candidates = await collection.find({
                "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
            }).limit(100).to_list(length=100)
            
            for candidate in candidates:
                candidate_goal = candidate.get("original_user_goal", "")
                normalized_candidate = " ".join(candidate_goal.lower().split())
                
                if normalized_candidate == normalized_goal:
                    logger.info(f"Cache NORMALIZED HIT on raw prompt (instant): {candidate.get('cache_id')}")
                    return {
                        "hit": True,
                        "cache_id": candidate.get("cache_id"),
                        "confidence": 1.0,
                        "match_method": "exact_normalized",
                        "cached_plan": candidate,
                    }
        
        return result
        
    except Exception as e:
        logger.warning(f"Exact cache lookup failed (continuing): {e}")
        return result


async def lookup_cached_plan(
    db,
    user_goal: str,
    canonical_goal: str,
    target_domain: Optional[str] = None,
    goal_embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Look up cached plan using three strategies:
    
    1. Exact match on canonical_goal (confidence=1.0)
    2. Keyword match: same domain + keyword overlap >= 0.6 (confidence=overlap_score)
    3. Semantic match: embedding cosine similarity >= 0.85 (confidence=similarity)
    
    Only return plans with avg_completion_rate >= MIN_COMPLETION_RATE.
    
    Returns:
        Dict with keys: hit, cache_id, confidence, match_method, cached_plan
    """
    result: Dict[str, Any] = {
        "hit": False,
        "cache_id": None,
        "confidence": 0.0,
        "match_method": "none",
        "cached_plan": None,
    }
    
    try:
        collection = db.plan_cache
        
        # Strategy 1: Exact match on canonical_goal
        exact_match = await collection.find_one({
            "canonical_goal": canonical_goal,
            "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
        })
        
        if exact_match:
            logger.info(f"Cache EXACT HIT: {exact_match.get('cache_id')}")
            return {
                "hit": True,
                "cache_id": exact_match.get("cache_id"),
                "confidence": 1.0,
                "match_method": "exact",
                "cached_plan": exact_match,
            }
        
        # Strategy 2: Keyword match with same domain
        goal_keywords = _extract_keywords(user_goal)
        
        if goal_keywords and target_domain:
            # Find candidates with same domain and good completion rate
            domain_candidates = await collection.find({
                "target_domain": target_domain,
                "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
            }).sort([
                ("avg_completion_rate", -1),
                ("total_uses", -1),
            ]).limit(20).to_list(length=20)
            
            best_keyword_match = None
            best_keyword_score = 0.0
            
            for candidate in domain_candidates:
                candidate_keywords = candidate.get("goal_keywords", [])
                similarity = _calculate_keyword_similarity(goal_keywords, candidate_keywords)
                
                if similarity >= MIN_KEYWORD_MATCH_RATIO and similarity > best_keyword_score:
                    best_keyword_score = similarity
                    best_keyword_match = candidate
            
            if best_keyword_match:
                logger.info(
                    f"Cache KEYWORD HIT: {best_keyword_match.get('cache_id')} "
                    f"score={best_keyword_score:.2f}"
                )
                return {
                    "hit": True,
                    "cache_id": best_keyword_match.get("cache_id"),
                    "confidence": best_keyword_score,
                    "match_method": "keyword",
                    "cached_plan": best_keyword_match,
                }
        
        # Strategy 3: Semantic match using embeddings
        if goal_embedding and len(goal_embedding) > 0:
            # Get candidates with embeddings
            # Note: For large scale, you'd want to use MongoDB Atlas Vector Search
            # or a dedicated vector DB. This is a simple in-memory approach.
            candidates = await collection.find({
                "goal_embedding": {"$exists": True, "$ne": []},
                "avg_completion_rate": {"$gte": MIN_COMPLETION_RATE},
            }).sort([
                ("avg_completion_rate", -1),
                ("total_uses", -1),
            ]).limit(50).to_list(length=50)
            
            best_semantic_match = None
            best_semantic_score = 0.0
            
            for candidate in candidates:
                candidate_embedding = candidate.get("goal_embedding", [])
                if candidate_embedding:
                    similarity = _cosine_similarity(goal_embedding, candidate_embedding)
                    
                    if similarity >= MIN_SIMILARITY_THRESHOLD and similarity > best_semantic_score:
                        best_semantic_score = similarity
                        best_semantic_match = candidate
            
            if best_semantic_match:
                logger.info(
                    f"Cache SEMANTIC HIT: {best_semantic_match.get('cache_id')} "
                    f"similarity={best_semantic_score:.2f}"
                )
                return {
                    "hit": True,
                    "cache_id": best_semantic_match.get("cache_id"),
                    "confidence": best_semantic_score,
                    "match_method": "semantic",
                    "cached_plan": best_semantic_match,
                }
        
        logger.info(f"Cache MISS for goal: {user_goal[:50]}...")
        return result
        
    except Exception as e:
        logger.warning(f"Cache lookup failed (continuing without cache): {e}")
        return result


async def save_plan_to_cache(
    db,
    session_id: str,
    user_goal: str,
    canonical_goal: str,
    planned_steps: List[Any],
    target_domain: Optional[str] = None,
    target_url: Optional[str] = None,
    goal_embedding: Optional[List[float]] = None,
) -> str:
    """
    Save a plan to cache.
    
    - Generate embedding if not provided
    - Extract keywords from user_goal
    - Set expires_at to CACHE_TTL_DAYS from now
    
    Returns:
        cache_id of the new cache entry
    """
    cache_id = str(uuid4())
    now = _utcnow()
    expires_at = now + timedelta(days=CACHE_TTL_DAYS)
    
    # Extract keywords
    goal_keywords = _extract_keywords(user_goal)
    
    # Generate embedding if not provided
    if not goal_embedding:
        try:
            goal_embedding = await embed_text(user_goal)
        except EmbeddingsError as e:
            logger.warning(f"Failed to generate embedding for cache: {e}")
            goal_embedding = []
    
    # Convert PlannedStep objects to dicts if needed
    steps_as_dicts = []
    for step in planned_steps:
        if hasattr(step, "model_dump"):
            steps_as_dicts.append(step.model_dump())
        elif isinstance(step, dict):
            steps_as_dicts.append(step)
        else:
            steps_as_dicts.append(dict(step))
    
    cache_doc = {
        "cache_id": cache_id,
        "canonical_goal": canonical_goal,
        "goal_keywords": goal_keywords,
        "goal_embedding": goal_embedding,
        "target_domain": target_domain,
        "target_url": target_url,
        "page_context": None,
        "planned_steps": steps_as_dicts,
        "total_steps": len(steps_as_dicts),
        "success_count": 1,  # Starting with 1 since it was just used successfully
        "failure_count": 0,
        "total_uses": 1,
        "avg_completion_rate": 1.0,  # 100% on first use
        "user_corrections": [],
        "created_at": now,
        "updated_at": now,
        "last_used_at": now,
        "expires_at": expires_at,
        "original_session_id": session_id,
        "original_user_goal": user_goal,
    }
    
    try:
        await db.plan_cache.insert_one(cache_doc)
        logger.info(f"Saved plan to cache: {cache_id} for goal: {user_goal[:50]}...")
        return cache_id
    except Exception as e:
        logger.warning(f"Failed to save plan to cache: {e}")
        return ""


async def record_cache_usage(
    db,
    cache_id: str,
    success: bool,
    session_id: Optional[str] = None,
) -> None:
    """
    Record cache usage and update metrics.
    
    - Increment success_count or failure_count
    - Increment total_uses
    - Recalculate avg_completion_rate
    - Update last_used_at
    """
    now = _utcnow()
    
    try:
        # First, get current values
        cache_doc = await db.plan_cache.find_one({"cache_id": cache_id})
        if not cache_doc:
            logger.warning(f"Cache entry not found for recording usage: {cache_id}")
            return
        
        current_success = cache_doc.get("success_count", 0)
        current_failure = cache_doc.get("failure_count", 0)
        
        if success:
            new_success = current_success + 1
            new_failure = current_failure
        else:
            new_success = current_success
            new_failure = current_failure + 1
        
        new_total = new_success + new_failure
        new_avg_rate = new_success / new_total if new_total > 0 else 0.0
        
        update_doc = {
            "$set": {
                "success_count": new_success,
                "failure_count": new_failure,
                "total_uses": new_total,
                "avg_completion_rate": new_avg_rate,
                "last_used_at": now,
                "updated_at": now,
            }
        }
        
        await db.plan_cache.update_one({"cache_id": cache_id}, update_doc)
        logger.info(
            f"Recorded cache usage: {cache_id} success={success} "
            f"new_rate={new_avg_rate:.2f}"
        )
        
    except Exception as e:
        logger.warning(f"Failed to record cache usage: {e}")


async def record_cache_correction(
    db,
    cache_id: str,
    step_number: int,
    correction: Dict[str, Any],
) -> None:
    """
    Record user correction for a cached plan.
    
    Append to user_corrections array with timestamp.
    """
    now = _utcnow()
    
    correction_record = {
        "step_number": step_number,
        "correction": correction,
        "timestamp": now,
    }
    
    try:
        await db.plan_cache.update_one(
            {"cache_id": cache_id},
            {
                "$push": {"user_corrections": correction_record},
                "$set": {"updated_at": now},
            }
        )
        logger.info(f"Recorded correction for cache: {cache_id} step={step_number}")
        
    except Exception as e:
        logger.warning(f"Failed to record cache correction: {e}")


async def invalidate_cache_entry(db, cache_id: str) -> None:
    """Delete a cache entry."""
    try:
        result = await db.plan_cache.delete_one({"cache_id": cache_id})
        if result.deleted_count > 0:
            logger.info(f"Invalidated cache entry: {cache_id}")
        else:
            logger.warning(f"Cache entry not found for invalidation: {cache_id}")
    except Exception as e:
        logger.warning(f"Failed to invalidate cache entry: {e}")


async def cleanup_low_quality_cache(
    db,
    min_uses: int = 5,
    max_failure_rate: float = 0.7,
) -> int:
    """
    Remove entries with total_uses >= min_uses AND avg_completion_rate < (1 - max_failure_rate).
    
    In other words, remove plans that have been tried enough times but have a high failure rate.
    
    Returns:
        Count of deleted entries
    """
    min_completion_rate = 1.0 - max_failure_rate  # 0.3 if max_failure_rate is 0.7
    
    try:
        result = await db.plan_cache.delete_many({
            "total_uses": {"$gte": min_uses},
            "avg_completion_rate": {"$lt": min_completion_rate},
        })
        
        deleted_count = result.deleted_count
        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} low-quality cache entries "
                f"(min_uses={min_uses}, min_rate={min_completion_rate:.2f})"
            )
        return deleted_count
        
    except Exception as e:
        logger.warning(f"Failed to cleanup low-quality cache: {e}")
        return 0


async def get_cache_stats(db) -> CacheStats:
    """
    Get statistics about the cache.
    """
    try:
        collection = db.plan_cache
        
        # Total entries
        total_entries = await collection.count_documents({})
        
        # Aggregate stats
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_uses": {"$sum": "$total_uses"},
                    "total_successes": {"$sum": "$success_count"},
                    "avg_completion_rate": {"$avg": "$avg_completion_rate"},
                }
            }
        ]
        
        agg_result = await collection.aggregate(pipeline).to_list(length=1)
        
        total_hits = 0
        avg_rate = 0.0
        
        if agg_result:
            total_hits = agg_result[0].get("total_successes", 0)
            avg_rate = agg_result[0].get("avg_completion_rate", 0.0) or 0.0
        
        # Entries by domain
        domain_pipeline = [
            {"$match": {"target_domain": {"$ne": None}}},
            {"$group": {"_id": "$target_domain", "count": {"$sum": 1}}},
        ]
        
        domain_result = await collection.aggregate(domain_pipeline).to_list(length=100)
        entries_by_domain = {item["_id"]: item["count"] for item in domain_result}
        
        return CacheStats(
            total_entries=total_entries,
            total_hits=total_hits,
            total_misses=0,  # Not tracked at this level
            hit_rate=0.0,  # Would need request-level tracking
            avg_completion_rate=avg_rate,
            entries_by_domain=entries_by_domain,
            entries_by_match_method={},
        )
        
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")
        return CacheStats()
