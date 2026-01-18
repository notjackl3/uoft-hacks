"""
FastAPI Routes for Commerce: Healthy Snack Recommendations & Buying Flow

Provides endpoints for:
- User profile management
- Healthy snack alternative recommendations
- Shopify buying flow orchestration
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.graph import graph_service
from app.services.shopify_catalog import ingest_shopify_catalog
from app.services.recommendation import recommend_healthy_alternatives
from app.services.orchestration import plan_buy_action, confirm_purchase

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateProfileRequest(BaseModel):
    user_id: str
    age_group: Optional[str] = "unknown"  # "minor", "adult", "unknown"
    preferences: Optional[List[Dict[str, str]]] = None  # [{type: "allergy", value: "peanuts"}]


class ProfileResponse(BaseModel):
    id: str
    user_id: str
    age_group: str
    preferences: List[Dict[str, Any]]


class UpdatePreferencesRequest(BaseModel):
    preferences: List[Dict[str, str]]  # [{type: "allergy", value: "peanuts"}, {type: "budget", value: "20 CAD"}]
    replace_existing: bool = False


class RecommendSnacksRequest(BaseModel):
    baseline_product_id: Optional[str] = None
    baseline_description: Optional[str] = None
    user_id: Optional[str] = None
    allergies: Optional[List[str]] = None
    diet: Optional[str] = None  # "vegetarian", "vegan", etc.
    budget: Optional[float] = None
    limit: int = 5


class RecommendationResponse(BaseModel):
    baseline: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    preferences_applied: Dict[str, Any]
    allergy_warning: Optional[str]
    decision_ids: List[str]
    session_id: str


class BuyRequest(BaseModel):
    product_id: str
    cart_session_id: Optional[str] = None
    current_step: Optional[str] = None
    ui_context: Dict[str, Any] = {}


class BuyConfirmRequest(BaseModel):
    product_id: str
    session_id: str
    confirmation_token: str
    ui_context: Dict[str, Any] = {}


class IngestCatalogRequest(BaseModel):
    store_url: str
    access_token: Optional[str] = None
    max_products: int = 250


# ============================================================================
# User Profile Endpoints
# ============================================================================

@router.post("/{company_id}/profile", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_profile(company_id: str, request: CreateProfileRequest):
    """
    Create or update a user profile with preferences.
    
    Preferences can include:
    - allergy: e.g., "peanuts", "dairy", "gluten"
    - diet: e.g., "vegetarian", "vegan"
    - budget: e.g., "20 CAD"
    - taste: e.g., "sweet", "savory"
    - calorie_limit: e.g., "200"
    - sugar_limit: e.g., "10g"
    """
    # Check if company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Check if profile already exists
    existing_profile = graph_service.get_user_profile(request.user_id)
    
    if existing_profile:
        # Clear existing preferences and add new ones
        graph_service.clear_user_preferences(request.user_id)
    else:
        # Create new profile
        profile = graph_service.create_user_profile(
            user_id=request.user_id,
            age_group=request.age_group
        )
        if not profile:
            raise HTTPException(status_code=500, detail="Failed to create profile")
    
    # Add preferences
    if request.preferences:
        for pref in request.preferences:
            graph_service.add_user_preference(
                user_id=request.user_id,
                pref_type=pref.get("type", "unknown"),
                value=pref.get("value", "")
            )
    
    # Return updated profile
    updated_profile = graph_service.get_user_profile(request.user_id)
    
    return ProfileResponse(
        id=updated_profile["id"],
        user_id=updated_profile["user_id"],
        age_group=updated_profile.get("age_group", "unknown"),
        preferences=updated_profile.get("preferences", [])
    )


@router.get("/{company_id}/profile/{user_id}")
async def get_profile(company_id: str, user_id: str):
    """Get a user's profile with preferences."""
    profile = graph_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.put("/{company_id}/profile/{user_id}/preferences")
async def update_preferences(company_id: str, user_id: str, request: UpdatePreferencesRequest):
    """Update user preferences."""
    profile = graph_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if request.replace_existing:
        graph_service.clear_user_preferences(user_id)
    
    for pref in request.preferences:
        graph_service.add_user_preference(
            user_id=user_id,
            pref_type=pref.get("type", "unknown"),
            value=pref.get("value", "")
        )
    
    return graph_service.get_user_profile(user_id)


# ============================================================================
# Healthy Snack Recommendations
# ============================================================================

@router.post("/{company_id}/recommendations/snacks", response_model=RecommendationResponse)
async def recommend_snacks(company_id: str, request: RecommendSnacksRequest):
    """
    Get healthy snack alternative recommendations.
    
    Provide either:
    - baseline_product_id: ID of a product in the catalog
    - baseline_description: Text description of the snack to find alternatives for
    
    The system will:
    1. Identify products in the same category
    2. Filter by allergies, diet, and budget preferences
    3. Score by health metrics (lower sugar, higher fiber/protein, whole ingredients)
    4. Return top recommendations with transparent justification
    
    SAFETY NOTES:
    - Recommendations focus on general health (lower added sugar, higher fiber/protein)
    - No weight-loss or restrictive dieting content
    - Allergy warning included if allergies not specified
    - For minors, recommendations are kept general
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Need either product ID or description
    if not request.baseline_product_id and not request.baseline_description:
        raise HTTPException(
            status_code=400,
            detail="Must provide either baseline_product_id or baseline_description"
        )
    
    # Build preferences dict
    preferences = {}
    if request.allergies:
        preferences["allergies"] = request.allergies
    if request.diet:
        preferences["diet"] = request.diet
    if request.budget:
        preferences["budget"] = request.budget
    
    try:
        result = await recommend_healthy_alternatives(
            company_id=company_id,
            baseline_product_id=request.baseline_product_id,
            baseline_description=request.baseline_description,
            user_id=request.user_id,
            preferences=preferences,
            limit=request.limit
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return RecommendationResponse(
            baseline=result["baseline"],
            recommendations=result["recommendations"],
            preferences_applied=result["preferences_applied"],
            allergy_warning=result.get("allergy_warning"),
            decision_ids=result["decision_ids"],
            session_id=result["session_id"]
        )
    
    except Exception as e:
        logger.exception(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Buying Flow
# ============================================================================

@router.post("/{company_id}/buy")
async def execute_buy_step(company_id: str, request: BuyRequest):
    """
    Execute the next step in the buying flow.
    
    Navigates through:
    1. Search product
    2. Open product page
    3. Add to cart
    4. View cart
    5. Proceed to checkout
    6. Enter shipping
    7. Continue to payment
    8. STOP: Review order (requires explicit confirmation)
    
    SAFETY: Will NEVER auto-complete a purchase. Always stops at review step.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Build session state
    session_id = request.cart_session_id or str(uuid.uuid4())
    session_state = {
        "session_id": session_id,
        "cart_id": session_id,
        "current_purchase_step": request.current_step,
    }
    
    try:
        result = await plan_buy_action(
            company_id=company_id,
            product_id=request.product_id,
            ui_context=request.ui_context,
            session_state=session_state
        )
        
        # Add session_id to result
        result["session_id"] = session_id
        
        # If confirmation required, provide the confirmation token format
        if result.get("requires_confirmation"):
            result["confirmation_required"] = True
            result["confirmation_endpoint"] = f"/api/commerce/{company_id}/buy/confirm"
            result["confirmation_token_format"] = f"CONFIRM_{session_id}_{request.product_id}"
        
        return result
    
    except Exception as e:
        logger.exception(f"Buy step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{company_id}/buy/confirm")
async def confirm_buy(company_id: str, request: BuyConfirmRequest):
    """
    Confirm a purchase after user review.
    
    SAFETY: Requires explicit confirmation token that matches the session/product.
    This endpoint only records the confirmation - actual purchase submission
    would need additional implementation.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        result = await confirm_purchase(
            company_id=company_id,
            product_id=request.product_id,
            session_id=request.session_id,
            confirmation_token=request.confirmation_token,
            ui_context=request.ui_context
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except Exception as e:
        logger.exception(f"Confirm purchase error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Catalog Ingestion
# ============================================================================

@router.post("/{company_id}/catalog/ingest")
async def ingest_catalog(company_id: str, request: IngestCatalogRequest):
    """
    Ingest a Shopify store's product catalog.
    
    Fetches products and stores them with:
    - Basic info (title, price, vendor, tags)
    - Parsed nutrition facts (if in description)
    - Embeddings for semantic search
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        result = await ingest_shopify_catalog(
            company_id=company_id,
            store_url=request.store_url,
            access_token=request.access_token,
            max_products=request.max_products
        )
        return result
    
    except Exception as e:
        logger.exception(f"Catalog ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Decision Trace
# ============================================================================

@router.get("/{company_id}/decisions/{decision_id}")
async def get_decision(company_id: str, decision_id: str):
    """
    Get a decision trace with all its justifications.
    
    Returns the decision with links to:
    - Recommended products
    - Comparisons
    - Evidence (nutrition claims, web sources)
    - Procedures followed
    """
    decision = graph_service.get_decision_trace(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision


@router.get("/{company_id}/products")
async def list_products(company_id: str, limit: int = 50):
    """List products for a company."""
    products = graph_service.find_products_by_category(company_id, limit=limit)
    return {"products": products, "count": len(products)}


@router.get("/{company_id}/products/{product_id}")
async def get_product(company_id: str, product_id: str):
    """Get a product with nutrition claims and evidence."""
    product = graph_service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
