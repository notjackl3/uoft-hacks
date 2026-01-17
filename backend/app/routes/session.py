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
from app.services.unified_planner import unified_plan, infer_target_fast

logger = logging.getLogger(__name__)
router = APIRouter()


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
    OPTIMIZED: Uses unified_planner for single LLM call that returns:
    - canonical_goal, target_url, target_domain
    - task_outline (phases)
    - first_step
    
    This replaces the previous 2-call pattern (normalize_goal + select_step).
    """
    now = _utcnow()
    session_id = str(uuid4())
    current_domain = _domain_from_url(request.url)

    # Fast heuristic check for target URL/domain (no LLM needed)
    fast_target_url, fast_target_domain = infer_target_fast(request.user_goal)
    
    # If target domain is known and we're not on it, return navigation step immediately
    if fast_target_domain and current_domain and not _domain_matches(fast_target_domain, current_domain):
        url_line = fast_target_url or f"https://{fast_target_domain}/"
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
        
        task_outline = ["Navigate to the website", "Complete the goal"]

        session_doc = {
            "session_id": session_id,
            "user_goal": request.user_goal,
            "canonical_goal": request.user_goal,
            "target_domain": fast_target_domain,
            "target_url": fast_target_url,
            "goal_embedding": [],
            "domain": current_domain,
            "planned_domain": fast_target_domain,
            "url": request.url,
            "last_seen_url": request.url,
            "last_seen_sig": _features_signature(request.initial_page_features),
            "mode": "single_step",
            "planned_steps": [wait_step.model_dump()],
            "current_step_number": 1,
            "last_sent_step_number": 1,
            "status": "in_progress",
            # New fields for progress tracking
            "task_outline": task_outline,
            "current_phase": 0,
            "completed_phases": [],
            "step_repeat_count": 0,  # For stuck detection
            "last_step_description": "",
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
            task_outline=task_outline,
            current_phase=0,
        )

    # UNIFIED PLANNER: Single LLM call for goal normalization + task outline + first step
    try:
        plan_result = await unified_plan(
            user_goal=request.user_goal,
            url=request.url,
            page_title=request.page_title,
            page_features=request.initial_page_features,
        )
    except Exception as e:
        logger.exception("Unified planner failed, falling back to legacy path")
        # Fallback to legacy normalize + select pattern
        norm = infer_target_from_goal(request.user_goal)
        if not norm.target_domain:
            try:
                norm = await normalize_goal_llm(request.user_goal)
            except Exception:
                pass
        
        canonical_goal = getattr(norm, "canonical_goal", None) or request.user_goal
        try:
            first_step = await select_next_step(
                step_number=1,
                canonical_goal=canonical_goal,
                url=request.url,
                page_title=request.page_title,
                page_features=request.initial_page_features,
                recent_steps=None,
            )
        except Exception as step_e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Step selector error: {step_e}") from step_e
        
        plan_result = type('PlanResult', (), {
            'canonical_goal': canonical_goal,
            'target_url': norm.target_url,
            'target_domain': norm.target_domain,
            'task_outline': ["Complete the task"],
            'current_phase': 0,
            'first_step': first_step,
        })()

    # Check if we need to navigate first (unified planner might have detected this)
    if plan_result.target_domain and current_domain and not _domain_matches(plan_result.target_domain, current_domain):
        url_line = plan_result.target_url or f"https://{plan_result.target_domain}/"
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
        first_step = wait_step
    else:
        first_step = plan_result.first_step

    planned_steps = [first_step]
    task_outline = plan_result.task_outline or ["Complete the task"]
    current_phase = plan_result.current_phase or 0

    try:
        goal_embedding = await embed_text(request.user_goal)
    except EmbeddingsError:
        goal_embedding = []  # Non-fatal, continue without embedding

    session_doc = {
        "session_id": session_id,
        "user_goal": request.user_goal,
        "canonical_goal": plan_result.canonical_goal,
        "target_domain": plan_result.target_domain,
        "target_url": plan_result.target_url,
        "goal_embedding": goal_embedding,
        "domain": current_domain,
        "planned_domain": current_domain,
        "url": request.url,
        "last_seen_url": request.url,
        "last_seen_sig": _features_signature(request.initial_page_features),
        "mode": "single_step",
        "planned_steps": [s.model_dump() for s in planned_steps],
        "current_step_number": first_step.step_number,
        "last_sent_step_number": first_step.step_number,
        "status": "in_progress",
        # New fields for progress tracking
        "task_outline": task_outline,
        "current_phase": current_phase,
        "completed_phases": [],
        "step_repeat_count": 0,  # For stuck detection
        "last_step_description": first_step.description or "",
        "created_at": now,
        "updated_at": now,
    }

    try:
        await db.sessions.insert_one(session_doc)
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"MongoDB error: {e}") from e

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
        total_steps=len(task_outline),  # Use outline length as total steps estimate
        first_step=first_step_payload,
        task_outline=task_outline,
        current_phase=current_phase,
    )


@router.post("/next", response_model=NextActionResponse)
async def next_action(request: NextActionRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    ENHANCED with progress tracking and stuck detection:
    1. Get session from MongoDB
    2. Advance step (if last sent step was executed successfully)
    3. Track progress against task_outline (phases)
    4. Detect stuck state (3+ same step without progress)
    5. Match current step to page_features
    6. If no match, try Gemini fallback
    7. Log execution
    8. Return instruction + target + progress info
    """
    session_doc = await db.sessions.find_one({"session_id": request.session_id})
    if not session_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Extract progress tracking fields
    task_outline: List[str] = session_doc.get("task_outline") or []
    current_phase: int = int(session_doc.get("current_phase", 0) or 0)
    completed_phases: List[int] = session_doc.get("completed_phases") or []
    step_repeat_count: int = int(session_doc.get("step_repeat_count", 0) or 0)
    last_step_description: str = str(session_doc.get("last_step_description") or "")

    total_steps = len(session_doc.get("planned_steps") or [])
    if total_steps == 0:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Session has no planned steps")

    if session_doc.get("status") == "completed":
        step_number = int(session_doc.get("current_step_number", total_steps))
        return NextActionResponse(
            step_number=step_number,
            total_steps=len(task_outline) or total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction="Session complete.",
            text_input=None,
            confidence=1.0,
            expected_page_change=False,
            session_complete=True,
            task_outline=task_outline,
            current_phase=current_phase,
            completed_phases=completed_phases,
        )

    now = _utcnow()
    current_step_number = int(session_doc.get("current_step_number", 1))
    last_sent = int(session_doc.get("last_sent_step_number", current_step_number))
    last_seen_url = str(session_doc.get("last_seen_url") or "")
    last_seen_sig = str(session_doc.get("last_seen_sig") or "")
    
    # Progress tracking: if action succeeded, potentially advance phase
    if request.previous_action_result.success:
        # Reset stuck counter on success
        step_repeat_count = 0
        
        # Check if current phase is complete (simple heuristic: step succeeded)
        # Advance to next phase if we have more phases
        if current_phase < len(task_outline) - 1:
            # For now, advance phase when step succeeds and URL changes significantly
            curr_sig = _features_signature(request.page_features)
            url_changed = bool(last_seen_url and request.url and request.url != last_seen_url)
            sig_changed = bool(last_seen_sig and curr_sig and curr_sig != last_seen_sig)
            
            if url_changed or sig_changed:
                if current_phase not in completed_phases:
                    completed_phases.append(current_phase)
                current_phase += 1
                logger.info(f"Phase advanced to {current_phase}: {task_outline[current_phase] if current_phase < len(task_outline) else 'DONE'}")

    # Update seen markers and progress tracking fields
    if request.url:
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "last_seen_url": request.url,
                    "last_seen_sig": _features_signature(request.page_features),
                    "current_phase": current_phase,
                    "completed_phases": completed_phases,
                    "step_repeat_count": step_repeat_count,
                    "updated_at": now,
                }
            },
        )

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
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {"$set": {"current_step_number": current_step_number, "last_sent_step_number": current_step_number, "updated_at": now}},
        )
        return NextActionResponse(
            step_number=step.step_number,
            total_steps=len(task_outline) or step.step_number,
            action=step.action,
            target_feature_index=None,
            target_feature=None,
            instruction=_instruction_for_step(step),
            text_input=None,
            confidence=1.0,
            expected_page_change=True,
            session_complete=False,
            task_outline=task_outline,
            current_phase=current_phase,
            completed_phases=completed_phases,
        )

    # In single_step mode, we never "run out" of steps; we keep generating.
    if mode != "single_step" and current_step_number > total_steps:
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {"$set": {"status": "completed", "updated_at": now, "current_step_number": total_steps}},
        )
        return NextActionResponse(
            step_number=total_steps,
            total_steps=len(task_outline) or total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction="Done.",
            text_input=None,
            confidence=1.0,
            expected_page_change=False,
            session_complete=True,
            task_outline=task_outline,
            current_phase=len(task_outline) - 1 if task_outline else 0,
            completed_phases=list(range(len(task_outline))) if task_outline else [],
        )

    step = _step_from_session(session_doc, current_step_number)
    if mode == "single_step" and (step is None):
        # Generate the next step on-demand and append it to planned_steps
        # Include task_outline context for better completion detection
        try:
            new_step = await select_next_step(
                step_number=current_step_number,
                canonical_goal=canonical_goal,
                url=request.url or session_doc.get("url", ""),
                page_title=request.page_title or "",
                page_features=request.page_features,
                recent_steps=[PlannedStep.model_validate(s) for s in (session_doc.get("planned_steps") or [])],
                task_outline=task_outline,
                current_phase=current_phase,
                completed_phases=completed_phases,
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Step selector error: {e}") from e

        # STUCK DETECTION: Check if same step description is repeated 3+ times
        new_desc = (new_step.description or "").strip().lower()
        if new_desc and new_desc == last_step_description.strip().lower():
            step_repeat_count += 1
            logger.warning(f"Step repeated {step_repeat_count} times: {new_desc[:50]}...")
        else:
            step_repeat_count = 0
            last_step_description = new_step.description or ""

        # If stuck (3+ repeats), try to recover
        if step_repeat_count >= 3:
            logger.warning(f"STUCK DETECTED: Same step repeated {step_repeat_count} times. Attempting recovery.")
            # Recovery: Try to generate a different step by asking LLM to suggest alternative
            try:
                recovery_step = await select_next_step(
                    step_number=current_step_number,
                    canonical_goal=f"[STUCK - try alternative approach] {canonical_goal}",
                    url=request.url or session_doc.get("url", ""),
                    page_title=request.page_title or "",
                    page_features=request.page_features,
                    recent_steps=[PlannedStep.model_validate(s) for s in (session_doc.get("planned_steps") or [])],
                    task_outline=task_outline,
                    current_phase=current_phase,
                    completed_phases=completed_phases,
                )
                new_step = recovery_step
                step_repeat_count = 0
                logger.info("Recovery step generated successfully")
            except Exception as recovery_e:
                logger.exception(f"Recovery failed: {recovery_e}")
                # If recovery fails, return a DONE with error message
                return NextActionResponse(
                    step_number=current_step_number,
                    total_steps=len(task_outline) or total_steps,
                    action="DONE",
                    target_feature_index=None,
                    target_feature=None,
                    instruction="Task appears stuck. The required element may not be on this page.",
                    text_input=None,
                    confidence=0.0,
                    expected_page_change=False,
                    session_complete=False,
                    task_outline=task_outline,
                    current_phase=current_phase,
                    completed_phases=completed_phases,
                )

        steps_list = session_doc.get("planned_steps") or []
        steps_list.append(new_step.model_dump())
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "planned_steps": steps_list,
                    "current_step_number": current_step_number,
                    "last_sent_step_number": current_step_number,
                    "step_repeat_count": step_repeat_count,
                    "last_step_description": last_step_description,
                    "updated_at": now,
                }
            },
        )
        step = new_step
        total_steps = len(steps_list)
    elif not step:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Current step not found")

    # Default no-target actions
    if step.action in {"SCROLL", "WAIT"}:
        match = {"matched": True, "feature_index": None, "confidence": 1.0, "feature": None}
        match_method = "algorithm"
    elif step.action == "DONE":
        # Mark all phases as complete
        all_phases = list(range(len(task_outline))) if task_outline else []
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": now,
                    "current_step_number": current_step_number,
                    "last_sent_step_number": current_step_number,
                    "current_phase": len(task_outline) - 1 if task_outline else 0,
                    "completed_phases": all_phases,
                }
            },
        )
        return NextActionResponse(
            step_number=current_step_number,
            total_steps=len(task_outline) or total_steps,
            action="DONE",
            target_feature_index=None,
            target_feature=None,
            instruction=step.description or "Done.",
            text_input=None,
            confidence=1.0,
            expected_page_change=step.expected_page_change,
            session_complete=True,
            task_outline=task_outline,
            current_phase=len(task_outline) - 1 if task_outline else 0,
            completed_phases=all_phases,
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
                    await db.sessions.update_one(
                        {"session_id": request.session_id},
                        {
                            "$set": {
                                "planned_steps": [s.model_dump() for s in new_steps],
                                "current_step_number": new_steps[0].step_number,
                                "last_sent_step_number": new_steps[0].step_number,
                                "updated_at": now,
                                "url": request.url,
                                "domain": _domain_from_url(request.url),
                                "planned_domain": _domain_from_url(request.url),
                            }
                        },
                    )

                    step = new_steps[0]
                    if step.action in {"SCROLL", "WAIT"}:
                        match = {"matched": True, "feature_index": None, "confidence": 1.0, "feature": None}
                    elif step.action == "DONE":
                        return NextActionResponse(
                            step_number=step.step_number,
                            total_steps=len(task_outline) or len(new_steps),
                            action="DONE",
                            target_feature_index=None,
                            target_feature=None,
                            instruction=step.description or "Done.",
                            text_input=None,
                            confidence=1.0,
                            expected_page_change=step.expected_page_change,
                            session_complete=True,
                            task_outline=task_outline,
                            current_phase=len(task_outline) - 1 if task_outline else 0,
                            completed_phases=list(range(len(task_outline))) if task_outline else [],
                        )
                    else:
                        match = match_element_to_step(step, request.page_features)
                        if not match["matched"]:
                            match = await fallback_to_gemini(step, request.page_features)

                    return NextActionResponse(
                        step_number=step.step_number,
                        total_steps=len(task_outline) or len(new_steps),
                        action=step.action,
                        target_feature_index=match["feature_index"],
                        target_feature=match["feature"],
                        instruction=_instruction_for_step(step),
                        text_input=step.text_input,
                        confidence=float(match["confidence"]),
                        expected_page_change=step.expected_page_change,
                        session_complete=False,
                        task_outline=task_outline,
                        current_phase=current_phase,
                        completed_phases=completed_phases,
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

    await db.sessions.update_one(
        {"session_id": request.session_id},
        {
            "$set": {
                "current_step_number": current_step_number,
                "last_sent_step_number": step.step_number,
                "current_phase": current_phase,
                "completed_phases": completed_phases,
                "updated_at": now,
            }
        },
    )

    return NextActionResponse(
        step_number=step.step_number,
        total_steps=len(task_outline) or total_steps,
        action=step.action,
        target_feature_index=match["feature_index"],
        target_feature=match["feature"],
        instruction=_instruction_for_step(step),
        text_input=step.text_input,
        confidence=float(match["confidence"]),
        expected_page_change=step.expected_page_change,
        session_complete=False,
        task_outline=task_outline,
        current_phase=current_phase,
        completed_phases=completed_phases,
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
    return {"ok": True, "step_number": step_number, "message": "Feedback recorded"}


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(session_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    session_doc = await db.sessions.find_one({"session_id": session_id})
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

