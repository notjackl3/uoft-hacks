"""
Admin endpoints for cache management.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

try:
    from motor.motor_asyncio import AsyncIOMotorDatabase
except Exception:
    AsyncIOMotorDatabase = Any  # type: ignore[misc,assignment]

from app.database import get_db
from app.models.cache import CacheStats
from app.services.cache_service import (
    cleanup_low_quality_cache,
    get_cache_stats,
    invalidate_cache_entry,
    lookup_cached_plan,
)
from app.services.embeddings import embed_text, EmbeddingsError
from app.services.goal_normalizer import infer_target_from_goal, normalize_goal_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats", response_model=CacheStats)
async def get_stats(db: AsyncIOMotorDatabase = Depends(get_db)) -> CacheStats:
    """
    Get cache statistics.
    
    Returns:
        - total_entries: Number of cached plans
        - total_hits: Total successful uses
        - avg_completion_rate: Average success rate across all cached plans
        - entries_by_domain: Count of entries per domain
    """
    return await get_cache_stats(db)


@router.delete("/{cache_id}")
async def delete_cache_entry(
    cache_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    Manually delete a cache entry.
    
    Args:
        cache_id: The unique identifier of the cache entry to delete
        
    Returns:
        Success status
    """
    await invalidate_cache_entry(db, cache_id)
    return {"ok": True, "message": f"Cache entry {cache_id} deleted"}


@router.post("/cleanup")
async def trigger_cleanup(
    min_uses: int = Query(default=5, ge=1, description="Minimum uses before considering for cleanup"),
    max_failure_rate: float = Query(default=0.7, ge=0.0, le=1.0, description="Maximum failure rate threshold"),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    Trigger cleanup of low-quality cache entries.
    
    Removes entries that have been used at least `min_uses` times
    but have a failure rate above `max_failure_rate`.
    
    Args:
        min_uses: Minimum number of uses before an entry can be cleaned up
        max_failure_rate: Entries with failure rate above this threshold will be removed
        
    Returns:
        Count of deleted entries
    """
    deleted_count = await cleanup_low_quality_cache(
        db,
        min_uses=min_uses,
        max_failure_rate=max_failure_rate,
    )
    return {
        "ok": True,
        "deleted_count": deleted_count,
        "criteria": {
            "min_uses": min_uses,
            "max_failure_rate": max_failure_rate,
        },
    }


@router.get("/lookup")
async def test_lookup(
    goal: str = Query(..., description="The user goal to look up"),
    domain: Optional[str] = Query(default=None, description="Target domain (e.g., 'shopify')"),
    include_embedding: bool = Query(default=False, description="Generate and use embedding for semantic search"),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    Test cache lookup without creating a session.
    
    Useful for debugging and testing cache behavior.
    
    Args:
        goal: The user goal to search for
        domain: Optional target domain to filter by
        include_embedding: Whether to generate embedding for semantic search
        
    Returns:
        Cache lookup result
    """
    # Normalize goal
    norm = infer_target_from_goal(goal)
    canonical_goal = goal  # Default to raw goal
    
    if not norm.target_domain:
        try:
            norm = await normalize_goal_llm(goal)
            canonical_goal = norm.canonical_goal or goal
        except Exception as e:
            logger.warning(f"Goal normalization failed: {e}")
    
    # Override domain if explicitly provided
    target_domain = domain or norm.target_domain
    
    # Generate embedding if requested
    goal_embedding = None
    if include_embedding:
        try:
            goal_embedding = await embed_text(goal)
        except EmbeddingsError as e:
            logger.warning(f"Embedding generation failed: {e}")
    
    # Perform lookup
    result = await lookup_cached_plan(
        db=db,
        user_goal=goal,
        canonical_goal=canonical_goal,
        target_domain=target_domain,
        goal_embedding=goal_embedding,
    )
    
    # Format response
    response = {
        "hit": result["hit"],
        "cache_id": result["cache_id"],
        "confidence": result["confidence"],
        "match_method": result["match_method"],
        "input": {
            "goal": goal,
            "canonical_goal": canonical_goal,
            "target_domain": target_domain,
            "embedding_used": goal_embedding is not None,
        },
    }
    
    if result["cached_plan"]:
        cached = result["cached_plan"]
        response["cached_plan_summary"] = {
            "original_goal": cached.get("original_user_goal"),
            "total_steps": cached.get("total_steps"),
            "success_count": cached.get("success_count"),
            "failure_count": cached.get("failure_count"),
            "avg_completion_rate": cached.get("avg_completion_rate"),
            "created_at": str(cached.get("created_at")),
            "last_used_at": str(cached.get("last_used_at")),
        }
    
    return response


@router.get("/entries")
async def list_cache_entries(
    domain: Optional[str] = Query(default=None, description="Filter by target domain"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of entries to return"),
    skip: int = Query(default=0, ge=0, description="Number of entries to skip"),
    sort_by: str = Query(default="avg_completion_rate", description="Field to sort by"),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    List cache entries with optional filtering.
    
    Args:
        domain: Filter by target domain
        limit: Maximum entries to return
        skip: Entries to skip (for pagination)
        sort_by: Field to sort by (avg_completion_rate, total_uses, created_at)
        
    Returns:
        List of cache entries
    """
    try:
        collection = db.plan_cache
        
        # Build query
        query = {}
        if domain:
            query["target_domain"] = domain
        
        # Build sort
        sort_field = sort_by if sort_by in ["avg_completion_rate", "total_uses", "created_at", "updated_at"] else "avg_completion_rate"
        sort_direction = -1  # Descending
        
        # Execute query
        cursor = collection.find(query).sort(sort_field, sort_direction).skip(skip).limit(limit)
        entries = await cursor.to_list(length=limit)
        
        # Get total count
        total_count = await collection.count_documents(query)
        
        # Format entries
        formatted_entries = []
        for entry in entries:
            formatted_entries.append({
                "cache_id": entry.get("cache_id"),
                "canonical_goal": entry.get("canonical_goal"),
                "original_user_goal": entry.get("original_user_goal"),
                "target_domain": entry.get("target_domain"),
                "total_steps": entry.get("total_steps"),
                "success_count": entry.get("success_count"),
                "failure_count": entry.get("failure_count"),
                "total_uses": entry.get("total_uses"),
                "avg_completion_rate": entry.get("avg_completion_rate"),
                "created_at": str(entry.get("created_at")),
                "last_used_at": str(entry.get("last_used_at")),
                "corrections_count": len(entry.get("user_corrections", [])),
            })
        
        return {
            "entries": formatted_entries,
            "total_count": total_count,
            "skip": skip,
            "limit": limit,
        }
        
    except Exception as e:
        logger.error(f"Failed to list cache entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cache entries: {str(e)}",
        )


@router.get("/entry/{cache_id}")
async def get_cache_entry(
    cache_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    """
    Get a specific cache entry by ID.
    
    Args:
        cache_id: The unique identifier of the cache entry
        
    Returns:
        Full cache entry details
    """
    try:
        entry = await db.plan_cache.find_one({"cache_id": cache_id})
        
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cache entry not found: {cache_id}",
            )
        
        # Remove MongoDB _id field
        entry.pop("_id", None)
        
        # Convert datetime fields to strings
        for field in ["created_at", "updated_at", "last_used_at", "expires_at"]:
            if entry.get(field):
                entry[field] = str(entry[field])
        
        return entry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache entry: {str(e)}",
        )
