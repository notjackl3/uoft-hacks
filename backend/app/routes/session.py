from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any

try:  # pragma: no cover
    from motor.motor_asyncio import AsyncIOMotorDatabase
except Exception:  # pragma: no cover
    AsyncIOMotorDatabase = Any  # type: ignore[misc,assignment]

from app.database import get_db
from app.models import (
    CorrectionRequest,
    NextActionRequest,
    NextActionResponse,
    PageFeature,
    PlannedStep,
    SessionStatusResponse,
    StartSessionRequest,
    StartSessionResponse,
    TargetHints,
)
from app.services.corrector import update_hints_from_actual_feature
from app.services.embeddings import EmbeddingsError, embed_text
from app.services.matcher import MatcherError, fallback_to_gemini, match_element_to_step
from app.services.planner import PlannerError, generate_workflow_plan
from app.services.goal_normalizer import infer_target_from_goal, normalize_goal_llm
from app.services.step_selector import select_next_step
from app.services.cache_service import (
    lookup_exact_match,
    lookup_cached_plan,
    save_plan_to_cache,
    record_cache_usage,
    record_cache_correction,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session cache for when MongoDB is unavailable
_session_cache: Dict[str, Dict[str, Any]] = {}


async def update_session(db, session_id: str, updates: Dict[str, Any]):
    """
    Update session in both MongoDB and cache
    Continues if MongoDB fails
    """
    try:
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": updates}
        )
    except Exception as e:
        logger.warning(f"MongoDB update failed for session {session_id}: {e}")
    
    # Always update cache
    if session_id in _session_cache:
        _session_cache[session_id].update(updates)


async def safe_db_operation(operation_name: str, db_call):
    """
    Wrapper to make MongoDB operations optional
    Logs failures but allows app to continue without database
    """
    try:
        await db_call()
        return True
    except Exception as e:
        logger.warning(f"MongoDB operation '{operation_name}' failed (continuing without persistence): {e}")
        return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host.lower()
    except Exception:
        return ""

def _same_domain(a: str, b: str) -> bool:
    return _domain_from_url(a) != "" and _domain_from_url(a) == _domain_from_url(b)

def _brand_key(domain: str) -> str:
    parts = [p for p in (domain or "").split(".") if p]
    if len(parts) < 2:
        return domain or ""
    return parts[-2]


def _domain_matches(target_domain: str, current_domain: str) -> bool:
    """
    More forgiving than exact match:
    - ignores www prefix
    - accepts subdomains
    - accepts same brand label (useful for amazon.ca vs amazon.com)
    """
    td = (target_domain or "").lower().lstrip(".")
    cd = (current_domain or "").lower().lstrip(".")
    if not td or not cd:
        return False

    td_no_www = td[4:] if td.startswith("www.") else td
    cd_no_www = cd[4:] if cd.startswith("www.") else cd

    if cd_no_www == td_no_www:
        return True
    if cd_no_www.endswith("." + td_no_www):
        return True
    return _brand_key(cd_no_www) == _brand_key(td_no_www)


def _features_signature(features: List[PageFeature]) -> str:
    # Small fingerprint to detect meaningful DOM/UI changes.
    parts: List[str] = []
    for f in (features or [])[:30]:
        parts.append(
            "|".join(
                [
                    f.type,
                    (f.text or "")[:50],
                    (f.placeholder or "")[:30],
                    (f.aria_label or "")[:30],
                    (f.href or "")[:50],
                ]
            )
        )
    return "||".join(parts)


def _step_from_session(session_doc: Dict[str, Any], step_number: int) -> Optional[PlannedStep]:
    steps = session_doc.get("planned_steps") or []
    for s in steps:
        if int(s.get("step_number", -1)) == int(step_number):
            return PlannedStep.model_validate(s)
    return None


def _replace_step_in_session(session_doc: Dict[str, Any], new_step: PlannedStep) -> List[Dict[str, Any]]:
    steps = session_doc.get("planned_steps") or []
    out: List[Dict[str, Any]] = []
    replaced = False
    for s in steps:
        if int(s.get("step_number", -1)) == int(new_step.step_number):
            out.append(new_step.model_dump())
            replaced = True
        else:
            out.append(s)
    if not replaced:
        out.append(new_step.model_dump())
    out.sort(key=lambda x: int(x.get("step_number", 0)))
    return out


def _instruction_for_step(step: PlannedStep) -> str:
    # Keep frontend-simple: just echo description; TYPE can include text_input if present.
    if step.action == "TYPE" and step.text_input:
        return f"{step.description} (text: {step.text_input})"
    return step.description


@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    1. Generate embedding for user_goal
    2. Call planner to get workflow steps
    3. Create session in MongoDB
    4. Match first step to initial_page_features
    5. Return session_id + first instruction
    """
    now = _utcnow()
    session_id = str(uuid4())

    # Goal normalization (cheap heuristic first; LLM only if uncertain).
    norm = infer_target_from_goal(request.user_goal)
    if not norm.target_domain:
        try:
            norm = await normalize_goal_llm(request.user_goal)
        except Exception:
            # If normalization fails, continue with raw goal and no target domain.
            pass

    # Credit-saving navigation gate: if we know the target domain and we're not on it,
    # return a manual URL step and do NOT call other LLMs yet.
    current_domain = _domain_from_url(request.url)
    if norm.target_domain and current_domain and not _domain_matches(norm.target_domain, current_domain):
        url_line = norm.target_url or f"https://{norm.target_domain}/"
        wait_step = PlannedStep(
            step_number=1,
            action="WAIT",
            description=(
                "Open a new tab, click the address bar, paste this URL, and press Enter:\n"
                f"{url_line}"
            ),
            target_hints=TargetHints(),
            expected_page_change=True,
        )

        session_doc = {
            "session_id": session_id,
            "user_goal": request.user_goal,
            "canonical_goal": norm.canonical_goal,
            "target_domain": norm.target_domain,
            "target_url": norm.target_url,
            "goal_embedding": [],
            "domain": current_domain,
            "planned_domain": norm.target_domain,
            "url": request.url,
            "last_seen_url": request.url,
            "last_seen_sig": _features_signature(request.initial_page_features),
            "mode": "single_step",
            "planned_steps": [wait_step.model_dump()],
            "current_step_number": 1,
            "last_sent_step_number": 1,
            "status": "in_progress",
            "created_at": now,
            "updated_at": now,
        }
        await db.sessions.insert_one(session_doc)

        return StartSessionResponse(
            session_id=session_id,
            planned_steps=[wait_step],
            total_steps=1,
            first_step={
                "step_number": 1,
                "action": "WAIT",
                "target_feature_index": None,
                "instruction": _instruction_for_step(wait_step),
                "confidence": 1.0,
            },
        )

    # Single-step selector mode: generate only the first step for the current page.
    canonical_goal = getattr(norm, "canonical_goal", None) or request.user_goal
    
    cache_id: Optional[str] = None
    cache_match_method: Optional[str] = None
    cache_confidence: float = 0.0
    goal_embedding: List[float] = []
    planned_steps: List[PlannedStep] = []
    
    # STEP 1: Try EXACT MATCH first - this is instant (no API calls)
    cache_result = await lookup_exact_match(
        db=db,
        canonical_goal=canonical_goal,
        user_goal=request.user_goal,  # Also try matching on raw prompt
    )
    
    if cache_result["hit"]:
        # INSTANT cache hit! No API calls needed at all
        cached_plan = cache_result["cached_plan"]
        planned_steps = [PlannedStep.model_validate(s) for s in cached_plan["planned_steps"]]
        
        cache_id = cache_result["cache_id"]
        cache_match_method = cache_result["match_method"]
        cache_confidence = cache_result["confidence"]
        
        # Use cached embedding if available (for session storage)
        goal_embedding = cached_plan.get("goal_embedding", [])
        
        logger.info(
            f"Cache EXACT HIT (instant, 0 API calls): {cache_id} "
            f"steps={len(planned_steps)}"
        )
    else:
        # STEP 2: No exact match - generate embedding for fuzzy matching
        try:
            goal_embedding = await embed_text(request.user_goal)
        except EmbeddingsError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Voyage AI error: {e}") from e
        
        # Try keyword + semantic matching
        cache_result = await lookup_cached_plan(
            db=db,
            user_goal=request.user_goal,
            canonical_goal=canonical_goal,
            target_domain=norm.target_domain,
            goal_embedding=goal_embedding,
        )
        
        if cache_result["hit"]:
            # Fuzzy cache hit (keyword or semantic)
            cached_plan = cache_result["cached_plan"]
            planned_steps = [PlannedStep.model_validate(s) for s in cached_plan["planned_steps"]]
            
            cache_id = cache_result["cache_id"]
            cache_match_method = cache_result["match_method"]
            cache_confidence = cache_result["confidence"]
            
            logger.info(
                f"Cache HIT ({cache_match_method}): {cache_id} "
                f"confidence={cache_confidence:.2f} steps={len(planned_steps)}"
            )
        else:
            # STEP 3: No cache hit at all - generate new plan via LLM
            try:
                first_step = await select_next_step(
                    step_number=1,
                    canonical_goal=canonical_goal,
                    url=request.url,
                    page_title=request.page_title,
                    page_features=request.initial_page_features,
                    recent_steps=None,
                )
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Step selector error: {e}") from e

            planned_steps = [first_step]
            logger.info("Cache MISS - generated new plan via LLM")

    # Safety net: for Instagram goals from the wrong site, ensure the first manual step includes the URL line
    # so the frontend can render it as a clickable link.
    try:
        if planned_steps and planned_steps[0].action == "WAIT":
            g = (request.user_goal or "").lower()
            if "instagram" in g and _domain_from_url(request.url) != "www.instagram.com":
                url_line = "https://www.instagram.com/accounts/emailsignup/"
                if url_line not in (planned_steps[0].description or ""):
                    planned_steps[0].description = (planned_steps[0].description or "").rstrip() + f"\n{url_line}"
                    planned_steps[0].expected_page_change = True
    except Exception:
        # Non-fatal: don't block session creation on string patching.
        pass

    total_steps = len(planned_steps)
    first_step = planned_steps[0]
    
    session_doc = {
        "session_id": session_id,
        "user_goal": request.user_goal,
        "canonical_goal": canonical_goal,
        "target_domain": norm.target_domain,
        "target_url": norm.target_url,
        "goal_embedding": goal_embedding,
        "domain": _domain_from_url(request.url),
        "planned_domain": _domain_from_url(request.url),
        "url": request.url,
        "last_seen_url": request.url,
        "last_seen_sig": _features_signature(request.initial_page_features),
        "mode": "single_step" if not cache_id else "cached",  # Track if using cache
        "planned_steps": [s.model_dump() for s in planned_steps],
        "current_step_number": first_step.step_number,
        "last_sent_step_number": first_step.step_number,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        # Cache tracking fields
        "cache_id": cache_id,
        "cache_match_method": cache_match_method,
        "cache_confidence": cache_confidence,
    }

    # Try to persist session, but continue even if MongoDB fails
    session_persisted = False
    try:
        await db.sessions.insert_one(session_doc)
        session_persisted = True
        logger.info(f"Session {session_id} persisted to MongoDB")
    except Exception as e:
        logger.warning(f"Failed to persist session to MongoDB (continuing without persistence): {e}")
        # Continue without database - session will work in memory only
    
    # Always cache in memory as backup
    _session_cache[session_id] = session_doc

    match = match_element_to_step(first_step, request.initial_page_features)
    match_method = "algorithm"
    if not match["matched"] and first_step.action in {"CLICK", "TYPE"}:
        try:
            match = await fallback_to_gemini(first_step, request.initial_page_features)
            match_method = "gemini_fallback"
        except MatcherError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Gemini matcher error: {e}") from e

    # Log step suggestion
    try:
        await db.execution_log.insert_one(
            {
                "session_id": session_id,
                "step_number": first_step.step_number,
                "planned_action": first_step.model_dump(),
                "page_features_received": [f.model_dump() for f in request.initial_page_features],
                "matched_feature": match["feature"].model_dump() if match["feature"] else None,
                "match_confidence": match["confidence"],
                "match_method": match_method,
                "user_feedback": None,
                "actual_feature_clicked": None,
                "timestamp": now,
            }
        )
    except Exception:
        logger.exception("Failed to insert execution log (start)")

    first_step_payload = {
        "step_number": first_step.step_number,
        "action": first_step.action,
        "target_feature_index": match["feature_index"],
        "instruction": _instruction_for_step(first_step),
        "confidence": match["confidence"],
    }

    return StartSessionResponse(
        session_id=session_id,
        planned_steps=planned_steps,
        total_steps=total_steps,
        first_step=first_step_payload,
    )


@router.post("/next", response_model=NextActionResponse)
async def next_action(request: NextActionRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    1. Get session from MongoDB or cache
    2. Advance step (if last sent step was executed successfully)
    3. Match current step to page_features
    4. If no match, try Gemini fallback
    5. Log execution
    6. Return instruction + target
    """
    # Try MongoDB first, fallback to cache
    session_doc = None
    try:
        session_doc = await db.sessions.find_one({"session_id": request.session_id})
    except Exception as e:
        logger.warning(f"MongoDB lookup failed, using cache: {e}")
    
    # Fallback to in-memory cache
    if not session_doc:
        session_doc = _session_cache.get(request.session_id)
        if not session_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        logger.info(f"Using cached session {request.session_id}")

    total_steps = len(session_doc.get("planned_steps") or [])
    if total_steps == 0:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Session has no planned steps")

    if session_doc.get("status") == "completed":
        step_number = int(session_doc.get("current_step_number", total_steps))
        return NextActionResponse(
            step_number=step_number,
            total_steps=total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction="Session complete.",
            text_input=None,
            confidence=1.0,
            expected_page_change=False,
            session_complete=True,
        )

    now = _utcnow()
    current_step_number = int(session_doc.get("current_step_number", 1))
    last_sent = int(session_doc.get("last_sent_step_number", current_step_number))
    last_seen_url = str(session_doc.get("last_seen_url") or "")
    last_seen_sig = str(session_doc.get("last_seen_sig") or "")

    # Update seen markers (we still use previous values below for change detection).
    if request.url:
        await update_session(db, request.session_id, {
            "last_seen_url": request.url,
            "last_seen_sig": _features_signature(request.page_features),
            "updated_at": now,
        })

    # NOTE: legacy full-plan "replan on domain change" removed.

    # Only advance if the client reports success for the last action we sent.
    if request.previous_action_result.success and last_sent == current_step_number:
        current_step_number += 1

    mode = session_doc.get("mode") or "planned"
    canonical_goal = session_doc.get("canonical_goal") or session_doc.get("user_goal") or ""
    target_domain = session_doc.get("target_domain")
    target_url = session_doc.get("target_url")

    # If we are in single_step mode and the user is still on the wrong domain, keep returning a manual WAIT step.
    if (
        mode == "single_step"
        and target_domain
        and request.url
        and not _domain_matches(target_domain, _domain_from_url(request.url))
    ):
        url_line = target_url or f"https://{target_domain}/"
        step = PlannedStep(
            step_number=current_step_number,
            action="WAIT",
            description=(
                "Open a new tab, click the address bar, paste this URL, and press Enter:\n"
                f"{url_line}"
            ),
            target_hints=TargetHints(),
            expected_page_change=True,
        )
        await update_session(db, request.session_id, {
            "current_step_number": current_step_number,
            "last_sent_step_number": current_step_number,
            "updated_at": now
        })
        return NextActionResponse(
            step_number=step.step_number,
            total_steps=step.step_number,
            action=step.action,
            target_feature_index=None,
            target_feature=None,
            instruction=_instruction_for_step(step),
            text_input=None,
            confidence=1.0,
            expected_page_change=True,
            session_complete=False,
        )

    # In single_step mode, we never "run out" of steps; we keep generating.
    if mode != "single_step" and current_step_number > total_steps:
        await update_session(db, request.session_id, {
            "status": "completed",
            "updated_at": now,
            "current_step_number": total_steps
        })
        return NextActionResponse(
            step_number=total_steps,
            total_steps=total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction="Done.",
            text_input=None,
            confidence=1.0,
            expected_page_change=False,
            session_complete=True,
        )

    step = _step_from_session(session_doc, current_step_number)
    if mode == "single_step" and (step is None):
        # Generate the next step on-demand and append it to planned_steps
        try:
            new_step = await select_next_step(
                step_number=current_step_number,
                canonical_goal=canonical_goal,
                url=request.url or session_doc.get("url", ""),
                page_title=request.page_title or "",
                page_features=request.page_features,
                recent_steps=[PlannedStep.model_validate(s) for s in (session_doc.get("planned_steps") or [])],
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Step selector error: {e}") from e

        steps_list = session_doc.get("planned_steps") or []
        steps_list.append(new_step.model_dump())
        await update_session(db, request.session_id, {
            "planned_steps": steps_list,
            "current_step_number": current_step_number,
            "last_sent_step_number": current_step_number,
            "updated_at": now,
        })
        step = new_step
        total_steps = len(steps_list)
    elif not step:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Current step not found")

    # Default no-target actions
    if step.action in {"SCROLL", "WAIT"}:
        match = {"matched": True, "feature_index": None, "confidence": 1.0, "feature": None}
        match_method = "algorithm"
    elif step.action == "DONE":
        await update_session(db, request.session_id, {
            "status": "completed",
            "updated_at": now,
            "current_step_number": current_step_number,
            "last_sent_step_number": current_step_number,
        })
        
        # Handle cache: record usage or save new plan
        cache_id = session_doc.get("cache_id")
        if cache_id:
            # Record successful cache usage
            await record_cache_usage(db, cache_id, success=True, session_id=request.session_id)
            logger.info(f"Recorded successful cache usage: {cache_id}")
        else:
            # Save this successful plan to cache for future reuse
            try:
                new_cache_id = await save_plan_to_cache(
                    db=db,
                    session_id=request.session_id,
                    user_goal=session_doc.get("user_goal", ""),
                    canonical_goal=session_doc.get("canonical_goal", ""),
                    planned_steps=[PlannedStep.model_validate(s) for s in session_doc.get("planned_steps", [])],
                    target_domain=session_doc.get("target_domain"),
                    target_url=session_doc.get("target_url"),
                    goal_embedding=session_doc.get("goal_embedding"),
                )
                if new_cache_id:
                    logger.info(f"Saved successful plan to cache: {new_cache_id}")
            except Exception as e:
                logger.warning(f"Failed to save plan to cache (non-fatal): {e}")
        
        return NextActionResponse(
            step_number=current_step_number,
            total_steps=total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction=step.description or "Done.",
            text_input=None,
            confidence=1.0,
            expected_page_change=step.expected_page_change,
            session_complete=True,
        )
    else:
        match = match_element_to_step(step, request.page_features)
        match_method = "algorithm"
        if not match["matched"]:
            try:
                match = await fallback_to_gemini(step, request.page_features)
                match_method = "gemini_fallback"
            except MatcherError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Gemini matcher error: {e}"
                ) from e

        # Adaptive behavior: if we can't match the current step to the current DOM,
        # and the page/DOM has changed since we last saw it, re-plan from the current page.
        if not match.get("matched", False) and request.url:
            curr_sig = _features_signature(request.page_features)
            url_changed = bool(last_seen_url and request.url and request.url != last_seen_url)
            sig_changed = bool(last_seen_sig and curr_sig and curr_sig != last_seen_sig)
            if url_changed or sig_changed:
                try:
                    new_steps = await generate_workflow_plan(
                        user_goal=session_doc.get("user_goal", ""),
                        initial_features=request.page_features,
                        url=request.url,
                        page_title=request.page_title or "",
                    )
                except PlannerError as e:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Planner error (replan): {e}"
                    ) from e

                if new_steps:
                    await update_session(db, request.session_id, {
                        "planned_steps": [s.model_dump() for s in new_steps],
                        "current_step_number": new_steps[0].step_number,
                        "last_sent_step_number": new_steps[0].step_number,
                        "updated_at": now,
                        "url": request.url,
                        "domain": _domain_from_url(request.url),
                        "planned_domain": _domain_from_url(request.url),
                    })

                    step = new_steps[0]
                    if step.action in {"SCROLL", "WAIT"}:
                        match = {"matched": True, "feature_index": None, "confidence": 1.0, "feature": None}
                    elif step.action == "DONE":
                        return NextActionResponse(
                            step_number=step.step_number,
                            total_steps=len(new_steps),
                            action="DONE",
                            target_feature_index=None,
                            target_feature=None,
                            instruction=step.description or "Done.",
                            text_input=None,
                            confidence=1.0,
                            expected_page_change=step.expected_page_change,
                            session_complete=True,
                        )
                    else:
                        match = match_element_to_step(step, request.page_features)
                        if not match["matched"]:
                            match = await fallback_to_gemini(step, request.page_features)

                    return NextActionResponse(
                        step_number=step.step_number,
                        total_steps=len(new_steps),
                        action=step.action,
                        target_feature_index=match["feature_index"],
                        target_feature=match["feature"],
                        instruction=_instruction_for_step(step),
                        text_input=step.text_input,
                        confidence=float(match["confidence"]),
                        expected_page_change=step.expected_page_change,
                        session_complete=False,
                    )

    # Log suggestion
    try:
        await db.execution_log.insert_one(
            {
                "session_id": request.session_id,
                "step_number": step.step_number,
                "planned_action": step.model_dump(),
                "page_features_received": [f.model_dump() for f in request.page_features],
                "matched_feature": match["feature"].model_dump() if match["feature"] else None,
                "match_confidence": match["confidence"],
                "match_method": match_method,
                "user_feedback": None,
                "actual_feature_clicked": None,
                "timestamp": now,
            }
        )
    except Exception:
        logger.exception("Failed to insert execution log (next)")

    await update_session(db, request.session_id, {
        "current_step_number": current_step_number,
        "last_sent_step_number": step.step_number,
        "updated_at": now,
    })

    return NextActionResponse(
        step_number=step.step_number,
        total_steps=total_steps,
        action=step.action,
        target_feature_index=match["feature_index"],
        target_feature=match["feature"],
        instruction=_instruction_for_step(step),
        text_input=step.text_input,
        confidence=float(match["confidence"]),
        expected_page_change=step.expected_page_change,
        session_complete=False,
    )


@router.post("/correct")
async def handle_correction(request: CorrectionRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    1. Log user correction
    2. Update target_hints in last-sent step
    3. Return updated step info
    """
    session_doc = await db.sessions.find_one({"session_id": request.session_id})
    if not session_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    step_number = int(session_doc.get("last_sent_step_number") or session_doc.get("current_step_number") or 1)
    step = _step_from_session(session_doc, step_number)
    if not step:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Step not found")

    if request.feedback == "wrong_element":
        if request.actual_feature_index is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="actual_feature_index required")

        # Find the most recent execution log entry for this step to locate the feature data.
        log_entry = await db.execution_log.find_one(
            {"session_id": request.session_id, "step_number": step_number},
            sort=[("timestamp", -1)],
        )
        if not log_entry:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No execution log found for step")

        features_raw = log_entry.get("page_features_received") or []
        features = [PageFeature.model_validate(f) for f in features_raw]
        actual = next((f for f in features if f.index == request.actual_feature_index), None)
        if actual is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="actual_feature_index not found in log")

        updated_hints = update_hints_from_actual_feature(step, actual)
        updated_step = step.model_copy(deep=True)
        updated_step.target_hints = updated_hints

        new_steps = _replace_step_in_session(session_doc, updated_step)
        now = _utcnow()

        await db.sessions.update_one(
            {"session_id": request.session_id},
            {"$set": {"planned_steps": new_steps, "updated_at": now}},
        )

        # Update log entry with feedback + actual clicked
        await db.execution_log.update_one(
            {"_id": log_entry["_id"]},
            {"$set": {"user_feedback": "wrong_element", "actual_feature_clicked": actual.model_dump()}},
        )
        
        # Also record correction for cache if this session used a cached plan
        cache_id = session_doc.get("cache_id")
        if cache_id:
            try:
                await record_cache_correction(
                    db=db,
                    cache_id=cache_id,
                    step_number=step_number,
                    correction={
                        "feedback": request.feedback,
                        "actual_feature_index": request.actual_feature_index,
                        "session_id": request.session_id,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to record cache correction (non-fatal): {e}")

        return {"ok": True, "step_number": step_number, "updated_target_hints": updated_hints.model_dump()}

    # doesn't_work: just log it for now
    now = _utcnow()
    await db.execution_log.insert_one(
        {
            "session_id": request.session_id,
            "step_number": step_number,
            "planned_action": step.model_dump(),
            "page_features_received": [],
            "matched_feature": None,
            "match_confidence": 0.0,
            "match_method": "algorithm",
            "user_feedback": "doesnt_work",
            "actual_feature_clicked": None,
            "timestamp": now,
        }
    )
    
    # Also record correction for cache if this session used a cached plan
    cache_id = session_doc.get("cache_id")
    if cache_id:
        try:
            await record_cache_correction(
                db=db,
                cache_id=cache_id,
                step_number=step_number,
                correction={
                    "feedback": request.feedback,
                    "session_id": request.session_id,
                },
            )
            # Record as a failure for cache metrics
            await record_cache_usage(db, cache_id, success=False, session_id=request.session_id)
        except Exception as e:
            logger.warning(f"Failed to record cache correction/failure (non-fatal): {e}")
    
    return {"ok": True, "step_number": step_number, "message": "Feedback recorded"}


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(session_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    # Try MongoDB first, fallback to cache
    session_doc = None
    try:
        session_doc = await db.sessions.find_one({"session_id": session_id})
    except Exception as e:
        logger.warning(f"MongoDB lookup failed: {e}")
    
    if not session_doc:
        session_doc = _session_cache.get(session_id)
    
    if not session_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    total_steps = len(session_doc.get("planned_steps") or [])
    return SessionStatusResponse(
        session_id=session_id,
        status=session_doc.get("status", "in_progress"),
        current_step_number=int(session_doc.get("current_step_number", 1)),
        total_steps=total_steps,
        user_goal=session_doc.get("user_goal", ""),
        url=session_doc.get("url", ""),
        updated_at=session_doc.get("updated_at") or _utcnow(),
    )

