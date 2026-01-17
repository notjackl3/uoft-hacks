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
from app.services.matcher import (
    MatcherError,
    fallback_to_gemini,
    match_element_to_step,
    filter_features_by_relevance,
)
from app.services.planner import PlannerError, generate_workflow_plan
from app.services.goal_normalizer import infer_target_from_goal, normalize_goal_llm
from app.services.step_selector import select_next_step

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


def _brand_key(domain: str) -> str:
    """
    Extract the main brand/company name from a hostname.
    
    Examples:
      - www.shopify.com -> shopify
      - admin.shopify.com -> shopify
      - accounts.google.com -> google
      - m.facebook.com -> facebook
      - amazon.co.uk -> amazon
    """
    if not domain:
        return ""
    
    domain = domain.lower().strip()
    
    # Remove protocol if present
    for prefix in ["https://", "http://", "//"]:
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
            break
    
    # Remove path/query/fragment
    domain = domain.split("/")[0].split("?")[0].split("#")[0]
    
    # Split into parts
    parts = [p for p in domain.split(".") if p]
    
    if not parts:
        return ""
    
    # Common prefixes to skip (subdomains)
    skip_prefixes = {
        "www", "m", "mobile", "admin", "app", "api", "accounts", 
        "login", "auth", "secure", "my", "portal", "dashboard"
    }
    
    # Common TLDs and country codes to skip
    skip_suffixes = {
        "com", "org", "net", "gov", "edu", "io", "ai", "app", "dev", "me", "tv", "co",
        "uk", "ca", "au", "de", "fr", "jp", "cn", "in", "br", "ru", "it", "es", "nl",
        "se", "no", "fi", "dk", "pl", "cz", "at", "ch", "be", "ie", "nz", "za", "mx",
        "ar", "cl", "kr", "sg", "hk", "tw", "th", "my", "ph", "id", "vn"
    }
    
    # Find the brand: skip prefixes from start, skip suffixes from end
    start = 0
    end = len(parts)
    
    # Skip known prefixes
    while start < end and parts[start] in skip_prefixes:
        start += 1
    
    # Skip known suffixes (TLDs, country codes)
    while end > start and parts[end - 1] in skip_suffixes:
        end -= 1
    
    # The brand should be the first remaining part
    if start < end:
        return parts[start]
    
    # Fallback: if everything was skipped, try the second-to-last part
    if len(parts) >= 2 and parts[-2] not in skip_prefixes:
        return parts[-2]
    
    # Last resort: return first part
    return parts[0] if parts else ""


def _domain_matches(target_domain: str, current_domain: str) -> bool:
    """
    Simple and flexible domain matching.
    Returns True if the brand name appears anywhere in either domain.
    
    Examples that MATCH:
    - "shopify.com" vs "admin.shopify.com" -> True
    - "shopify.com" vs "shopify.ca" -> True  
    - "instagram.com" vs "www.instagram.com" -> True
    - "google.com" vs "accounts.google.com" -> True
    - "amazon.com" vs "amazon.co.uk" -> True
    """
    if not target_domain or not current_domain:
        return False
    
    # Normalize both inputs
    td = target_domain.lower().strip()
    cd = current_domain.lower().strip()
    
    # Remove protocols
    for prefix in ["https://", "http://", "//"]:
        if td.startswith(prefix):
            td = td[len(prefix):]
        if cd.startswith(prefix):
            cd = cd[len(prefix):]
    
    # Remove paths, just keep hostname
    td = td.split("/")[0].split("?")[0].split("#")[0]
    cd = cd.split("/")[0].split("?")[0].split("#")[0]
    
    # Exact match
    if td == cd:
        return True
    
    # Extract brand names
    target_brand = _brand_key(td)
    current_brand = _brand_key(cd)
    
    if not target_brand:
        return False
    
    # Does the target brand appear anywhere in the current domain?
    if target_brand in cd:
        return True
    
    # Check reverse
    if current_brand and current_brand in td:
        return True
    
    return False


def _features_signature(features: List[PageFeature]) -> str:
    """Small fingerprint to detect meaningful DOM/UI changes."""
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
            pass

    # Credit-saving navigation gate: if we know the target domain and we're not on it,
    # return a manual URL step and do NOT call other LLMs yet.
    current_domain = _domain_from_url(request.url)
    target_brand = _brand_key(norm.target_domain) if norm.target_domain else ""
    
    # Simple check: does the target brand appear in the current URL?
    user_on_correct_site = target_brand and target_brand in request.url.lower()
    
    if norm.target_domain and not user_on_correct_site:
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

    try:
        goal_embedding = await embed_text(request.user_goal)
    except EmbeddingsError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Voyage AI error: {e}") from e

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
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Step selector error: {e}") from e

    planned_steps = [first_step]

    # Safety net: for Instagram goals from the wrong site
    try:
        if planned_steps and planned_steps[0].action == "WAIT":
            g = (request.user_goal or "").lower()
            if "instagram" in g and _domain_from_url(request.url) != "www.instagram.com":
                url_line = "https://www.instagram.com/accounts/emailsignup/"
                if url_line not in (planned_steps[0].description or ""):
                    planned_steps[0].description = (planned_steps[0].description or "").rstrip() + f"\n{url_line}"
                    planned_steps[0].expected_page_change = True
    except Exception:
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
        "mode": "single_step",
        "planned_steps": [s.model_dump() for s in planned_steps],
        "current_step_number": first_step.step_number,
        "last_sent_step_number": first_step.step_number,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
    }

    try:
        await db.sessions.insert_one(session_doc)
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"MongoDB error: {e}") from e

    # Filter features by relevance before matching
    filtered_features = filter_features_by_relevance(
        page_features=request.initial_page_features,
        current_step=first_step,
        goal=request.user_goal,
        min_confidence=0.3,
        max_features=50,
    )

    logger.info(
        f"[start_session] Filtered {len(request.initial_page_features)} -> {len(filtered_features)} features"
    )

    match = match_element_to_step(first_step, filtered_features)
    match_method = "algorithm"
    if not match["matched"] and first_step.action in {"CLICK", "TYPE"}:
        try:
            match = await fallback_to_gemini(first_step, filtered_features)
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
                "page_features_filtered": [f.model_dump() for f in filtered_features],
                "page_features_filtered_count": len(filtered_features),
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
    1. Get session from MongoDB
    2. Advance step (if last sent step was executed successfully)
    3. Match current step to page_features
    4. If no match, try Gemini fallback
    5. Log execution
    6. Return instruction + target
    """
    session_doc = await db.sessions.find_one({"session_id": request.session_id})
    if not session_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

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

    if request.url:
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "last_seen_url": request.url,
                    "last_seen_sig": _features_signature(request.page_features),
                    "updated_at": now,
                }
            },
        )

    if request.previous_action_result.success and last_sent == current_step_number:
        current_step_number += 1

    if request.force_advance:
        current_step_number = max(current_step_number, last_sent) + 1

    mode = session_doc.get("mode") or "planned"
    canonical_goal = session_doc.get("canonical_goal") or session_doc.get("user_goal") or ""
    target_domain = session_doc.get("target_domain")
    target_url = session_doc.get("target_url")

    # If we are in single_step mode and the user is still on the wrong domain
    if (
        mode == "single_step"
        and target_domain
        and request.url
        and not request.force_advance
    ):
        target_brand = _brand_key(target_domain)
        
        # Simple check: does the target brand appear in the current URL?
        user_on_correct_site = target_brand and target_brand in request.url.lower()
        
        if not user_on_correct_site:
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

    if mode != "single_step" and current_step_number > total_steps:
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {"$set": {"status": "completed", "updated_at": now, "current_step_number": total_steps}},
        )
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
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "planned_steps": steps_list,
                    "current_step_number": current_step_number,
                    "last_sent_step_number": current_step_number,
                    "updated_at": now,
                }
            },
        )
        step = new_step
        total_steps = len(steps_list)
    elif not step:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Current step not found")

    # Initialize filtered_features for logging
    filtered_features: List[PageFeature] = []

    if step.action in {"SCROLL", "WAIT"}:
        match = {"matched": True, "feature_index": None, "confidence": 1.0, "feature": None}
        match_method = "algorithm"
    elif step.action == "DONE":
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": now,
                    "current_step_number": current_step_number,
                    "last_sent_step_number": current_step_number,
                }
            },
        )
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
        # Filter features by relevance before matching
        filtered_features = filter_features_by_relevance(
            page_features=request.page_features,
            current_step=step,
            goal=canonical_goal,
            min_confidence=0.3,
            max_features=50,
        )

        logger.info(
            f"[next_action] Step {step.step_number}: Filtered {len(request.page_features)} -> {len(filtered_features)} features"
        )

        match = match_element_to_step(step, filtered_features)
        match_method = "algorithm"
        if not match["matched"]:
            try:
                match = await fallback_to_gemini(step, filtered_features)
                match_method = "gemini_fallback"
            except MatcherError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Gemini matcher error: {e}"
                ) from e

        # Adaptive behavior for re-planning
        if mode != "single_step" and (not match.get("matched", False)) and request.url:
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
                        # Re-filter for new step
                        filtered_features = filter_features_by_relevance(
                            page_features=request.page_features,
                            current_step=step,
                            goal=canonical_goal,
                            min_confidence=0.3,
                            max_features=50,
                        )
                        match = match_element_to_step(step, filtered_features)
                        if not match["matched"]:
                            match = await fallback_to_gemini(step, filtered_features)

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
                "page_features_filtered": [f.model_dump() for f in filtered_features] if filtered_features else [],
                "page_features_filtered_count": len(filtered_features) if filtered_features else 0,
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
                "updated_at": now,
            }
        },
    )

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

        await db.execution_log.update_one(
            {"_id": log_entry["_id"]},
            {"$set": {"user_feedback": "wrong_element", "actual_feature_clicked": actual.model_dump()}},
        )

        return {"ok": True, "step_number": step_number, "updated_target_hints": updated_hints.model_dump()}

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