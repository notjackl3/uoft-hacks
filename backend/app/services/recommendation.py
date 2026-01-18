"""
Recommendation Engine for Healthy Snack Alternatives

Graph-first recommendation with transparent justification.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.graph import graph_service
from app.services.doc_ingestion import embed_text
from app.services.web_search import search_healthier_alternatives, gather_product_evidence

logger = logging.getLogger(__name__)


# Snack categories for grouping
SNACK_CATEGORIES = {
    "chips": ["chips", "crisps", "crackers", "pretzels", "popcorn"],
    "cookies": ["cookies", "biscuits", "wafers"],
    "candy": ["candy", "chocolate", "sweets", "gummies"],
    "bars": ["bar", "granola bar", "protein bar", "energy bar"],
    "nuts": ["nuts", "almonds", "cashews", "peanuts", "mixed nuts", "trail mix"],
    "dried_fruit": ["dried fruit", "raisins", "dates", "figs"],
    "yogurt": ["yogurt", "yoghurt"],
    "other": [],
}


def categorize_product(product: Dict[str, Any]) -> str:
    """Determine snack category for a product."""
    title_lower = product.get("title", "").lower()
    tags = [t.lower() for t in product.get("tags", [])]
    product_type = (product.get("product_type") or "").lower()
    
    combined_text = f"{title_lower} {' '.join(tags)} {product_type}"
    
    for category, keywords in SNACK_CATEGORIES.items():
        if any(kw in combined_text for kw in keywords):
            return category
    
    return "other"


def filter_by_preferences(
    products: List[Dict[str, Any]],
    preferences: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Filter products based on user preferences.
    
    Returns (filtered_products, exclusion_reasons)
    """
    filtered = []
    reasons = []
    
    allergies = preferences.get("allergies", [])
    diet = preferences.get("diet")  # "vegetarian", "vegan", etc.
    budget = preferences.get("budget")  # Max price
    
    for product in products:
        excluded = False
        exclusion_reason = None
        
        # Check allergies
        if allergies:
            ingredients = (product.get("ingredients") or "").lower()
            description = (product.get("description") or "").lower()
            combined = f"{ingredients} {description}"
            
            for allergy in allergies:
                allergy_lower = allergy.lower()
                # Common allergy mappings
                allergy_keywords = {
                    "peanuts": ["peanut", "peanuts", "groundnut"],
                    "tree nuts": ["almond", "cashew", "walnut", "hazelnut", "pecan", "pistachio", "macadamia"],
                    "dairy": ["milk", "dairy", "lactose", "cheese", "butter", "cream", "whey"],
                    "gluten": ["wheat", "gluten", "barley", "rye", "oats"],
                    "soy": ["soy", "soya", "soybean"],
                    "eggs": ["egg", "eggs"],
                    "shellfish": ["shellfish", "shrimp", "crab", "lobster"],
                }
                
                keywords = allergy_keywords.get(allergy_lower, [allergy_lower])
                if any(kw in combined for kw in keywords):
                    excluded = True
                    exclusion_reason = f"Contains allergen: {allergy}"
                    break
        
        # Check diet
        if not excluded and diet:
            tags = [t.lower() for t in product.get("tags", [])]
            description = (product.get("description") or "").lower()
            
            if diet.lower() == "vegan":
                non_vegan = ["milk", "dairy", "egg", "honey", "gelatin", "whey"]
                if any(nv in description for nv in non_vegan):
                    if "vegan" not in tags:
                        excluded = True
                        exclusion_reason = f"May not be vegan"
            
            elif diet.lower() == "vegetarian":
                non_veg = ["gelatin", "lard", "tallow"]
                if any(nv in description for nv in non_veg):
                    excluded = True
                    exclusion_reason = "May contain animal products"
        
        # Check budget
        if not excluded and budget:
            price = product.get("price")
            if price and price > budget:
                excluded = True
                exclusion_reason = f"Price ${price} exceeds budget ${budget}"
        
        if excluded:
            reasons.append(f"{product.get('title', 'Unknown')}: {exclusion_reason}")
        else:
            filtered.append(product)
    
    return filtered, reasons


def calculate_health_score(product: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate a health score for a product.
    
    Returns (score 0-100, list of factors)
    
    Note: Health-focused suggestions only. We avoid weight-loss/restrictive dieting.
    Focus on: lower added sugar, higher fiber/protein, whole ingredients.
    """
    score = 50.0  # Base score
    factors = []
    
    # Get nutrition claims from product
    nutrition = product.get("nutrition_claims", [])
    nutrition_map = {n.get("metric"): n.get("value") for n in nutrition}
    
    # Sugar assessment (lower is better)
    sugar = nutrition_map.get("sugar_g")
    if sugar is not None:
        if sugar <= 5:
            score += 20
            factors.append("Low sugar (5g or less per serving)")
        elif sugar <= 10:
            score += 10
            factors.append("Moderate sugar (10g or less per serving)")
        elif sugar > 20:
            score -= 15
            factors.append("High sugar content")
    
    # Fiber assessment (higher is better)
    fiber = nutrition_map.get("fiber_g")
    if fiber is not None:
        if fiber >= 5:
            score += 20
            factors.append("High fiber (5g or more per serving)")
        elif fiber >= 3:
            score += 10
            factors.append("Good fiber content (3g+ per serving)")
    
    # Protein assessment (higher is better for snacks)
    protein = nutrition_map.get("protein_g")
    if protein is not None:
        if protein >= 10:
            score += 15
            factors.append("High protein (10g or more per serving)")
        elif protein >= 5:
            score += 10
            factors.append("Good protein content (5g+ per serving)")
    
    # Check tags/description for health indicators
    tags = [t.lower() for t in product.get("tags", [])]
    description = (product.get("description") or "").lower()
    combined = f"{' '.join(tags)} {description}"
    
    positive_indicators = [
        ("whole grain", 10, "Contains whole grains"),
        ("whole wheat", 10, "Made with whole wheat"),
        ("organic", 5, "Organic ingredients"),
        ("no added sugar", 15, "No added sugar"),
        ("sugar free", 12, "Sugar-free"),
        ("high fiber", 12, "High fiber"),
        ("high protein", 10, "High protein"),
        ("plant based", 8, "Plant-based"),
        ("natural", 5, "Natural ingredients"),
        ("no artificial", 8, "No artificial ingredients"),
    ]
    
    for indicator, points, factor_text in positive_indicators:
        if indicator in combined:
            score += points
            if factor_text not in factors:
                factors.append(factor_text)
    
    # Check for less healthy indicators
    negative_indicators = [
        ("highly processed", -10, "Highly processed"),
        ("artificial colors", -8, "Contains artificial colors"),
        ("artificial flavors", -5, "Contains artificial flavors"),
        ("high fructose corn syrup", -15, "Contains high fructose corn syrup"),
        ("hydrogenated", -12, "Contains hydrogenated oils"),
    ]
    
    for indicator, points, factor_text in negative_indicators:
        if indicator in combined:
            score += points
            factors.append(factor_text)
    
    # Clamp score
    score = max(0, min(100, score))
    
    # If no nutrition data, lower confidence
    if not nutrition:
        factors.insert(0, "⚠️ Limited nutrition data available")
        score = min(score, 60)  # Cap score without data
    
    return score, factors


def compare_products(
    baseline: Dict[str, Any],
    alternative: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a comparison between baseline and alternative products.
    
    Returns comparison summary with reasons.
    """
    baseline_score, baseline_factors = calculate_health_score(baseline)
    alt_score, alt_factors = calculate_health_score(alternative)
    
    reasons = []
    
    # Compare nutrition if available
    baseline_nutrition = {n.get("metric"): n.get("value") for n in baseline.get("nutrition_claims", [])}
    alt_nutrition = {n.get("metric"): n.get("value") for n in alternative.get("nutrition_claims", [])}
    
    # Sugar comparison
    if "sugar_g" in baseline_nutrition and "sugar_g" in alt_nutrition:
        sugar_diff = baseline_nutrition["sugar_g"] - alt_nutrition["sugar_g"]
        if sugar_diff > 0:
            reasons.append(f"{sugar_diff:.1f}g less sugar per serving")
    
    # Fiber comparison
    if "fiber_g" in baseline_nutrition and "fiber_g" in alt_nutrition:
        fiber_diff = alt_nutrition["fiber_g"] - baseline_nutrition["fiber_g"]
        if fiber_diff > 0:
            reasons.append(f"{fiber_diff:.1f}g more fiber per serving")
    
    # Protein comparison
    if "protein_g" in baseline_nutrition and "protein_g" in alt_nutrition:
        protein_diff = alt_nutrition["protein_g"] - baseline_nutrition["protein_g"]
        if protein_diff > 0:
            reasons.append(f"{protein_diff:.1f}g more protein per serving")
    
    # Add qualitative differences
    baseline_tags = set(t.lower() for t in baseline.get("tags", []))
    alt_tags = set(t.lower() for t in alternative.get("tags", []))
    
    health_tags = {"organic", "whole grain", "high fiber", "high protein", "vegan", "gluten free", "no added sugar"}
    alt_health_tags = alt_tags.intersection(health_tags) - baseline_tags
    if alt_health_tags:
        reasons.append(f"Additional benefits: {', '.join(alt_health_tags)}")
    
    # Price comparison
    baseline_price = baseline.get("price")
    alt_price = alternative.get("price")
    if baseline_price and alt_price:
        price_diff = baseline_price - alt_price
        if price_diff > 0:
            reasons.append(f"${price_diff:.2f} cheaper")
        elif price_diff < -1:
            reasons.append(f"${abs(price_diff):.2f} more expensive")
    
    is_healthier = alt_score > baseline_score
    
    return {
        "baseline_title": baseline.get("title"),
        "alternative_title": alternative.get("title"),
        "baseline_score": baseline_score,
        "alternative_score": alt_score,
        "score_improvement": alt_score - baseline_score,
        "is_healthier": is_healthier,
        "reasons": reasons if reasons else ["Similar nutritional profile"],
        "alternative_factors": alt_factors,
        "reason_summary": " • ".join(reasons[:3]) if reasons else "Similar to baseline"
    }


async def recommend_healthy_alternatives(
    company_id: str,
    baseline_product_id: Optional[str] = None,
    baseline_description: Optional[str] = None,
    user_id: Optional[str] = None,
    preferences: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recommend healthier snack alternatives.
    
    This is the main recommendation function.
    
    Safety note: If user age is unknown or minor, recommendations are kept general
    (lower added sugar, higher fiber/protein, whole ingredients) without promoting
    restrictive dieting or weight-loss content.
    """
    logger.info(f"Generating healthy alternatives for company {company_id}")
    
    preferences = preferences or {}
    session_id = session_id or str(uuid.uuid4())
    
    # Get user profile if available
    user_profile = None
    if user_id:
        user_profile = graph_service.get_user_profile(user_id)
        if user_profile:
            # Merge stored preferences with request preferences
            stored_prefs = {p["type"]: p["value"] for p in user_profile.get("preferences", [])}
            
            # Get allergies from stored preferences
            if not preferences.get("allergies") and "allergy" in stored_prefs:
                preferences["allergies"] = [stored_prefs["allergy"]]
            
            # Get diet preference
            if not preferences.get("diet") and "diet" in stored_prefs:
                preferences["diet"] = stored_prefs["diet"]
            
            # Get budget
            if not preferences.get("budget") and "budget" in stored_prefs:
                try:
                    preferences["budget"] = float(stored_prefs["budget"].replace("CAD", "").replace("$", "").strip())
                except ValueError:
                    pass
    
    # SAFETY: Check for allergies before proceeding
    # If recommending food and allergies not specified, we should note this
    allergy_warning = None
    if not preferences.get("allergies"):
        allergy_warning = "⚠️ No allergies specified. Please confirm any food allergies before purchasing."
    
    # Resolve baseline product
    baseline = None
    if baseline_product_id:
        baseline = graph_service.get_product(baseline_product_id)
    
    if not baseline and baseline_description:
        # Create a temporary baseline representation
        baseline = {
            "id": None,
            "title": baseline_description,
            "description": baseline_description,
            "tags": [],
            "nutrition_claims": [],
            "is_temporary": True
        }
    
    if not baseline:
        return {
            "error": "No baseline product provided",
            "recommendations": [],
            "decision_ids": []
        }
    
    baseline_category = categorize_product(baseline)
    baseline_score, baseline_factors = calculate_health_score(baseline)
    
    # Step 1: Candidate generation - Graph query
    category_tags = SNACK_CATEGORIES.get(baseline_category, [])
    candidates = graph_service.find_products_by_category(
        company_id=company_id,
        tags=category_tags if category_tags else None,
        limit=50
    )
    
    # Also try vector similarity
    if baseline.get("description"):
        baseline_embedding = embed_text(baseline.get("description", baseline.get("title", "")))
        similar_products = graph_service.find_similar_products(
            company_id=company_id,
            query_embedding=baseline_embedding,
            limit=30
        )
        
        # Merge candidates
        seen_ids = {p["id"] for p in candidates}
        for product in similar_products:
            if product["id"] not in seen_ids:
                candidates.append(product)
                seen_ids.add(product["id"])
    
    # Remove baseline from candidates
    if baseline.get("id"):
        candidates = [p for p in candidates if p["id"] != baseline["id"]]
    
    # Step 2: Filter by preferences (allergies, diet, budget)
    filtered_candidates, exclusion_reasons = filter_by_preferences(candidates, preferences)
    
    if exclusion_reasons:
        logger.info(f"Excluded {len(exclusion_reasons)} products due to preferences")
    
    # Step 3: Score and rank
    scored_candidates = []
    for product in filtered_candidates:
        # Get full product with nutrition
        full_product = graph_service.get_product(product["id"])
        if not full_product:
            full_product = product
        
        health_score, factors = calculate_health_score(full_product)
        
        # Only recommend if healthier than baseline
        if health_score > baseline_score:
            comparison = compare_products(baseline, full_product)
            
            scored_candidates.append({
                "product": full_product,
                "health_score": health_score,
                "factors": factors,
                "comparison": comparison,
                "score_improvement": health_score - baseline_score
            })
    
    # Sort by score improvement
    scored_candidates.sort(key=lambda x: x["score_improvement"], reverse=True)
    
    # Take top N
    top_recommendations = scored_candidates[:limit]
    
    # Step 4: Gather evidence for top recommendations
    evidence_ids = []
    comparison_ids = []
    
    for rec in top_recommendations:
        product = rec["product"]
        
        # Gather web evidence
        product_evidence_ids = await gather_product_evidence(
            product_id=product["id"],
            product_title=product.get("title", ""),
            company_id=company_id
        )
        evidence_ids.extend(product_evidence_ids)
        
        # Create comparison node if baseline has an ID
        if baseline.get("id"):
            comparison = graph_service.create_comparison(
                baseline_product_id=baseline["id"],
                alternative_product_id=product["id"],
                reason_summary=rec["comparison"]["reason_summary"]
            )
            if comparison:
                comparison_ids.append(comparison["id"])
    
    # Step 5: Create decision trace
    decision_ids = []
    if top_recommendations:
        recommended_product_ids = [r["product"]["id"] for r in top_recommendations]
        
        reasoning = f"Recommended {len(top_recommendations)} healthier alternatives to {baseline.get('title', 'unknown product')}. "
        reasoning += f"Baseline health score: {baseline_score:.0f}. "
        if preferences.get("allergies"):
            reasoning += f"Filtered for allergies: {', '.join(preferences['allergies'])}. "
        if preferences.get("diet"):
            reasoning += f"Diet preference: {preferences['diet']}. "
        
        decision = graph_service.create_recommendation_decision(
            session_id=session_id,
            user_id=user_id or "anonymous",
            baseline_product_id=baseline.get("id"),
            recommended_product_ids=recommended_product_ids,
            comparison_ids=comparison_ids,
            evidence_ids=evidence_ids,
            reasoning=reasoning
        )
        
        if decision:
            decision_ids.append(decision["id"])
    
    # Build response
    recommendations = []
    for rec in top_recommendations:
        product = rec["product"]
        recommendations.append({
            "product_id": product["id"],
            "title": product.get("title"),
            "vendor": product.get("vendor"),
            "price": product.get("price"),
            "currency": product.get("currency", "CAD"),
            "url": product.get("url"),
            "image_url": product.get("image_url"),
            "health_score": rec["health_score"],
            "score_improvement": rec["score_improvement"],
            "why_healthier": rec["factors"],
            "comparison_summary": rec["comparison"]["reason_summary"],
            "confidence": "high" if rec["health_score"] > 70 else ("medium" if rec["health_score"] > 50 else "low")
        })
    
    return {
        "baseline": {
            "title": baseline.get("title"),
            "health_score": baseline_score,
            "factors": baseline_factors
        },
        "recommendations": recommendations,
        "preferences_applied": preferences,
        "exclusion_reasons": exclusion_reasons[:5],  # Limit for brevity
        "allergy_warning": allergy_warning,
        "decision_ids": decision_ids,
        "session_id": session_id,
        "total_candidates": len(candidates),
        "filtered_candidates": len(filtered_candidates),
        "healthier_alternatives_found": len(top_recommendations)
    }
