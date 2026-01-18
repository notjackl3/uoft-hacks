"""
Orchestration Service for Doc-Following Navigation + Commerce Buying Flow

Provides runtime navigation planning based on ingested documentation procedures.
Includes commerce procedures for Shopify buying flow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.graph import graph_service
from app.services.doc_ingestion import embed_text

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def match_ui_state(step_expected_state: Optional[str], ui_context: Dict[str, Any]) -> float:
    """
    Match a step's expected state against current UI context.
    
    Returns a score between 0 and 1.
    """
    if not step_expected_state:
        return 0.5  # Neutral if no expected state
    
    expected_lower = step_expected_state.lower()
    
    # Check URL match
    current_url = ui_context.get("url", "").lower()
    if current_url:
        # URL contains expected keywords
        keywords = expected_lower.split()
        url_matches = sum(1 for kw in keywords if kw in current_url)
        if url_matches > 0:
            return min(1.0, 0.5 + url_matches * 0.2)
    
    # Check page title match
    page_title = ui_context.get("title", "").lower()
    if page_title and any(word in page_title for word in expected_lower.split()):
        return 0.8
    
    # Check visible text match
    visible_text = ui_context.get("visible_text", "").lower()
    if visible_text:
        keywords = expected_lower.split()
        text_matches = sum(1 for kw in keywords if kw in visible_text)
        if text_matches > 0:
            return min(1.0, 0.3 + text_matches * 0.1)
    
    # Check element hints
    elements = ui_context.get("elements", [])
    for elem in elements:
        elem_text = elem.get("text", "").lower()
        if any(word in elem_text for word in expected_lower.split()):
            return 0.7
    
    return 0.2  # Low match


def find_matching_step(
    procedure: Dict[str, Any],
    ui_context: Dict[str, Any],
    completed_step_ids: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Find the next step in a procedure that matches the current UI state.
    
    Returns the step with the best match, or None if no good match.
    """
    steps = procedure.get("steps", [])
    if not steps:
        return None
    
    best_step = None
    best_score = 0.0
    
    for step in steps:
        # Skip already completed steps
        if step.get("id") in completed_step_ids:
            continue
        
        # Check if previous step is completed (for sequential procedures)
        step_idx = step.get("step_index", 0)
        if step_idx > 1:
            prev_step_completed = any(
                s.get("id") in completed_step_ids 
                for s in steps 
                if s.get("step_index") == step_idx - 1
            )
            if not prev_step_completed and completed_step_ids:
                continue
        
        # Score this step
        state_score = match_ui_state(step.get("expected_state"), ui_context)
        
        # Boost score for first uncompleted step
        if not completed_step_ids and step_idx == 1:
            state_score += 0.3
        
        if state_score > best_score:
            best_score = state_score
            best_step = step
    
    return best_step if best_score > 0.3 else None


def build_action(step: Dict[str, Any], ui_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an action object from a step and current UI context.
    
    Returns: {type, selector/url, text, instruction, step_id}
    """
    action_type = step.get("action_type", "unknown")
    selector_hint = step.get("selector_hint")
    
    action = {
        "type": action_type,
        "instruction": step.get("instruction"),
        "step_id": step.get("id"),
    }
    
    if action_type == "click":
        # Try to find matching element in UI context
        selector = find_element_selector(selector_hint, ui_context)
        action["selector"] = selector
    
    elif action_type == "type":
        selector = find_element_selector(selector_hint, ui_context, prefer_input=True)
        action["selector"] = selector
        action["text"] = step.get("selector_hint", "")  # The hint might be what to type
    
    elif action_type == "navigate":
        action["url"] = selector_hint
    
    elif action_type == "wait":
        action["duration"] = 1000  # Default 1 second
    
    return action


def find_element_selector(
    hint: Optional[str],
    ui_context: Dict[str, Any],
    prefer_input: bool = False
) -> Optional[str]:
    """
    Find a CSS selector for an element matching the hint.
    
    Uses elements from UI context to find the best match.
    """
    if not hint:
        return None
    
    hint_lower = hint.lower()
    elements = ui_context.get("elements", [])
    
    best_match = None
    best_score = 0.0
    
    for elem in elements:
        elem_text = elem.get("text", "").lower()
        elem_type = elem.get("type", "").lower()
        elem_selector = elem.get("selector")
        
        # Skip non-matching types if preference set
        if prefer_input and elem_type not in ["input", "textarea", "select"]:
            continue
        
        # Score based on text match
        score = 0.0
        if hint_lower in elem_text:
            score = 0.8
        elif any(word in elem_text for word in hint_lower.split()):
            score = 0.5
        
        # Boost for exact match
        if elem_text == hint_lower:
            score = 1.0
        
        if score > best_score:
            best_score = score
            best_match = elem_selector
    
    return best_match


async def plan_next_ui_action(
    company_id: str,
    user_goal: str,
    ui_context: Dict[str, Any],
    session_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Plan the next UI action based on user goal and current UI state.
    
    Args:
        company_id: The company whose docs to use
        user_goal: What the user wants to accomplish
        ui_context: Current UI state {url, title, visible_text, elements[]}
        session_state: Optional session state {current_procedure_id, completed_step_ids}
    
    Returns:
        {
            procedure_id: str,
            step_id: str,
            action: {type, selector/url, text, instruction},
            confidence: float,
            justification: str
        }
    """
    logger.info(f"Planning action for company {company_id}, goal: {user_goal[:50]}...")
    
    session_state = session_state or {}
    current_procedure_id = session_state.get("current_procedure_id")
    completed_step_ids = session_state.get("completed_step_ids", [])
    
    # If we have a current procedure, try to continue with it
    if current_procedure_id:
        procedure = graph_service.get_procedure_with_steps(current_procedure_id)
        if procedure:
            step = find_matching_step(procedure, ui_context, completed_step_ids)
            if step:
                action = build_action(step, ui_context)
                
                # Record decision trace
                graph_service.create_decision(
                    session_id=session_state.get("session_id", "unknown"),
                    action_type=action["type"],
                    action_data=action,
                    procedure_id=current_procedure_id,
                    step_id=step["id"]
                )
                
                return {
                    "procedure_id": current_procedure_id,
                    "step_id": step["id"],
                    "action": action,
                    "confidence": 0.8,
                    "justification": f"Continuing procedure: {procedure.get('goal', 'Unknown')}"
                }
    
    # Find a matching procedure using embedding similarity
    goal_embedding = embed_text(user_goal)
    similar_procedures = graph_service.find_similar_procedures(
        company_id=company_id,
        goal_embedding=goal_embedding,
        limit=5
    )
    
    if not similar_procedures:
        # No procedures found - return exploratory action
        return {
            "procedure_id": None,
            "step_id": None,
            "action": {
                "type": "observe",
                "instruction": "No matching procedure found. Please explore the page or refine your goal."
            },
            "confidence": 0.0,
            "justification": "No documentation procedures match this goal"
        }
    
    # Try each procedure to find one with a matching step
    for proc_match in similar_procedures:
        procedure = proc_match["procedure"]
        similarity_score = proc_match["score"]
        
        # Get full procedure with steps
        full_procedure = graph_service.get_procedure_with_steps(procedure["id"])
        if not full_procedure:
            continue
        
        # Find matching step
        step = find_matching_step(full_procedure, ui_context, completed_step_ids)
        if step:
            action = build_action(step, ui_context)
            
            # Record decision trace
            graph_service.create_decision(
                session_id=session_state.get("session_id", "unknown"),
                action_type=action["type"],
                action_data=action,
                procedure_id=procedure["id"],
                step_id=step["id"]
            )
            
            return {
                "procedure_id": procedure["id"],
                "step_id": step["id"],
                "action": action,
                "confidence": similarity_score * 0.8,
                "justification": f"Following procedure: {procedure.get('goal', 'Unknown')}"
            }
    
    # No matching step found in any procedure
    # Return the first step of the best matching procedure
    if similar_procedures:
        best_procedure = similar_procedures[0]["procedure"]
        full_procedure = graph_service.get_procedure_with_steps(best_procedure["id"])
        
        if full_procedure and full_procedure.get("steps"):
            first_step = full_procedure["steps"][0]
            action = build_action(first_step, ui_context)
            
            return {
                "procedure_id": best_procedure["id"],
                "step_id": first_step["id"],
                "action": action,
                "confidence": 0.5,
                "justification": f"Starting procedure (UI state may not match): {best_procedure.get('goal', 'Unknown')}"
            }
    
    return {
        "procedure_id": None,
        "step_id": None,
        "action": {
            "type": "observe",
            "instruction": "Cannot determine next action. The current page may not match any known procedures."
        },
        "confidence": 0.0,
        "justification": "No matching procedure steps for current UI state"
    }


async def get_relevant_context(
    company_id: str,
    user_goal: str,
    ui_context: Dict[str, Any],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get relevant documentation chunks for the current context.
    
    Useful for providing context to an LLM for more complex reasoning.
    """
    goal_embedding = embed_text(user_goal)
    
    # Find similar chunks
    chunks = graph_service.find_similar_chunks(
        company_id=company_id,
        query_embedding=goal_embedding,
        limit=limit
    )
    
    # Score and sort by relevance to current UI
    for chunk in chunks:
        url_match = 0.2 if ui_context.get("url", "") in chunk.get("page_url", "") else 0
        chunk["relevance_boost"] = url_match
    
    return sorted(chunks, key=lambda x: x.get("relevance_boost", 0), reverse=True)


# ============================================================================
# Commerce Buying Flow
# ============================================================================

# Standard Shopify purchase steps
PURCHASE_STEPS = [
    {
        "name": "search_product",
        "instruction": "Search for the product using the search bar",
        "action_type": "type",
        "selector_hints": ["search", "search-input", "[type='search']", "[placeholder*='Search']"],
        "expected_state": "On store homepage or search results page",
        "requires_confirmation": False,
    },
    {
        "name": "open_product",
        "instruction": "Click on the product to open its page",
        "action_type": "click",
        "selector_hints": ["product-title", "product-link", ".product-card"],
        "expected_state": "Search results showing the product",
        "requires_confirmation": False,
    },
    {
        "name": "add_to_cart",
        "instruction": "Click the Add to Cart button",
        "action_type": "click",
        "selector_hints": ["add-to-cart", "AddToCart", "[name='add']", "button:contains('Add to Cart')"],
        "expected_state": "On product page with Add to Cart visible",
        "requires_confirmation": False,
    },
    {
        "name": "view_cart",
        "instruction": "Go to cart to review items",
        "action_type": "click",
        "selector_hints": ["cart", "cart-icon", ".cart-link", "a[href*='cart']"],
        "expected_state": "Product added confirmation shown",
        "requires_confirmation": False,
    },
    {
        "name": "proceed_checkout",
        "instruction": "Proceed to checkout",
        "action_type": "click",
        "selector_hints": ["checkout", "Checkout", "[name='checkout']", "button:contains('Checkout')"],
        "expected_state": "On cart page with items",
        "requires_confirmation": False,
    },
    {
        "name": "enter_shipping",
        "instruction": "Enter shipping information (email, address)",
        "action_type": "type",
        "selector_hints": ["email", "shipping-address", "[autocomplete='email']"],
        "expected_state": "On checkout shipping step",
        "requires_confirmation": False,
    },
    {
        "name": "continue_to_payment",
        "instruction": "Continue to payment step",
        "action_type": "click",
        "selector_hints": ["continue", "Continue to payment", "[data-step='payment']"],
        "expected_state": "Shipping info completed",
        "requires_confirmation": False,
    },
    {
        "name": "review_order",
        "instruction": "Review order details before final confirmation - STOP HERE AND REQUEST USER CONFIRMATION",
        "action_type": "wait",
        "selector_hints": [],
        "expected_state": "On payment/review page",
        "requires_confirmation": True,  # CRITICAL: Must stop here
    },
]


def get_purchase_step_by_name(step_name: str) -> Optional[Dict[str, Any]]:
    """Get a purchase step by its name."""
    for i, step in enumerate(PURCHASE_STEPS):
        if step["name"] == step_name:
            return {**step, "index": i}
    return None


def get_next_purchase_step(current_step_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get the next step after the current one."""
    if not current_step_name:
        return {**PURCHASE_STEPS[0], "index": 0}
    
    for i, step in enumerate(PURCHASE_STEPS):
        if step["name"] == current_step_name:
            if i + 1 < len(PURCHASE_STEPS):
                return {**PURCHASE_STEPS[i + 1], "index": i + 1}
    return None


def match_purchase_step_to_ui(step: Dict[str, Any], ui_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Match a purchase step to the current UI context.
    
    Returns action with best matching selector.
    """
    action = {
        "type": step["action_type"],
        "instruction": step["instruction"],
        "step_name": step["name"],
        "requires_confirmation": step.get("requires_confirmation", False),
    }
    
    elements = ui_context.get("elements", [])
    selector_hints = step.get("selector_hints", [])
    
    # Try to find matching element
    for hint in selector_hints:
        hint_lower = hint.lower()
        for elem in elements:
            elem_selector = elem.get("selector", "")
            elem_text = elem.get("text", "").lower()
            
            # Match by selector hint
            if hint_lower in elem_selector.lower():
                action["selector"] = elem_selector
                action["matched_element"] = elem
                return action
            
            # Match by text
            if hint_lower in elem_text:
                action["selector"] = elem_selector
                action["matched_element"] = elem
                return action
    
    # No match found - return without selector
    action["selector"] = None
    action["warning"] = "Could not find matching element on page"
    return action


async def plan_buy_action(
    company_id: str,
    product_id: str,
    ui_context: Dict[str, Any],
    session_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Plan the next action in the buying flow.
    
    SAFETY: Will STOP at review_order step and require explicit confirmation.
    
    Returns:
        {
            step_name: str,
            action: {type, selector, instruction},
            requires_confirmation: bool,
            is_complete: bool,
            justification: str,
            decision_id: str
        }
    """
    session_state = session_state or {}
    session_id = session_state.get("session_id", "unknown")
    current_step = session_state.get("current_purchase_step")
    cart_id = session_state.get("cart_id")
    
    # Get product info
    product = graph_service.get_product(product_id)
    if not product:
        return {
            "error": f"Product {product_id} not found",
            "step_name": None,
            "action": None,
            "requires_confirmation": False,
            "is_complete": False,
        }
    
    # Determine current step based on URL and UI
    current_url = ui_context.get("url", "").lower()
    
    # Infer step from URL
    inferred_step = None
    if "/cart" in current_url:
        inferred_step = "view_cart"
    elif "/checkout" in current_url:
        if "shipping" in current_url:
            inferred_step = "enter_shipping"
        elif "payment" in current_url:
            inferred_step = "review_order"
        else:
            inferred_step = "proceed_checkout"
    elif "/products/" in current_url:
        # Check if product is added (look for confirmation)
        visible_text = ui_context.get("visible_text", "").lower()
        if "added to cart" in visible_text or "added to your cart" in visible_text:
            inferred_step = "add_to_cart"  # Just completed this step
        else:
            inferred_step = "open_product"  # Just completed this step
    elif "/search" in current_url or "q=" in current_url:
        inferred_step = "search_product"  # Just completed this step
    
    # Use inferred step if no current step or if inferred is ahead
    if inferred_step:
        inferred_idx = get_purchase_step_by_name(inferred_step)
        current_idx = get_purchase_step_by_name(current_step) if current_step else None
        
        if current_idx is None or (inferred_idx and inferred_idx["index"] >= (current_idx.get("index", 0) if current_idx else 0)):
            current_step = inferred_step
    
    # Get next step
    next_step_data = get_next_purchase_step(current_step)
    
    if not next_step_data:
        return {
            "step_name": "complete",
            "action": None,
            "requires_confirmation": False,
            "is_complete": True,
            "justification": "Purchase flow completed (at review step)",
        }
    
    # Check if this is the confirmation step
    if next_step_data.get("requires_confirmation"):
        return {
            "step_name": next_step_data["name"],
            "action": {
                "type": "wait",
                "instruction": "⚠️ STOP: Please review your order and confirm you want to proceed with purchase.",
            },
            "requires_confirmation": True,
            "is_complete": False,
            "justification": "Reached review step - explicit user confirmation required before proceeding",
            "product": {
                "title": product.get("title"),
                "price": product.get("price"),
                "currency": product.get("currency", "CAD"),
            }
        }
    
    # Build action for next step
    action = match_purchase_step_to_ui(next_step_data, ui_context)
    
    # Special handling for type actions
    if action["type"] == "type" and next_step_data["name"] == "search_product":
        action["text"] = product.get("title", "")
    
    # Record decision trace
    decision = graph_service.create_decision(
        session_id=session_id,
        action_type=action["type"],
        action_data={
            "step_name": next_step_data["name"],
            "product_id": product_id,
            **action
        },
    )
    
    # Add product to cart in graph if at add_to_cart step
    if next_step_data["name"] == "add_to_cart" and cart_id:
        graph_service.add_item_to_cart(cart_id, product_id)
    
    return {
        "step_name": next_step_data["name"],
        "action": action,
        "requires_confirmation": False,
        "is_complete": False,
        "justification": f"Executing step: {next_step_data['instruction']}",
        "decision_id": decision["id"] if decision else None,
        "product": {
            "title": product.get("title"),
            "price": product.get("price"),
        }
    }


async def confirm_purchase(
    company_id: str,
    product_id: str,
    session_id: str,
    confirmation_token: str,
    ui_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Confirm and execute the final purchase step.
    
    SAFETY: Requires explicit confirmation token before proceeding.
    This should only be called after user has reviewed the order.
    
    In this implementation, we stop at review and don't auto-complete purchase.
    """
    # Validate confirmation token (in production, use proper token validation)
    expected_token = f"CONFIRM_{session_id}_{product_id}"
    
    if confirmation_token != expected_token:
        return {
            "error": "Invalid confirmation token",
            "status": "rejected",
            "message": "Confirmation token does not match. Please review your order again.",
        }
    
    # Get product info
    product = graph_service.get_product(product_id)
    if not product:
        return {
            "error": f"Product {product_id} not found",
            "status": "failed",
        }
    
    # Record the confirmed decision
    decision = graph_service.create_decision(
        session_id=session_id,
        action_type="purchase_confirmed",
        action_data={
            "product_id": product_id,
            "product_title": product.get("title"),
            "price": product.get("price"),
            "confirmed_by_user": True,
        }
    )
    
    # NOTE: In production, this would actually submit the order
    # For safety, we're just recording the confirmation
    
    return {
        "status": "confirmed",
        "message": "Purchase confirmed. In production, this would complete the order.",
        "product": {
            "title": product.get("title"),
            "price": product.get("price"),
            "currency": product.get("currency", "CAD"),
        },
        "decision_id": decision["id"] if decision else None,
        "confirmation_token": confirmation_token,
    }
