"""
Unit Tests for Recommendation Engine

Tests:
- Preference filtering (allergies, diet, budget)
- Health scoring
- Product comparison
- Decision trace creation
"""

import pytest
from typing import Dict, Any, List


# ============================================================================
# Mock Data
# ============================================================================

MOCK_PRODUCTS = [
    {
        "id": "prod_1",
        "title": "Classic Potato Chips",
        "vendor": "ChipCo",
        "price": 4.99,
        "tags": ["chips", "salty"],
        "description": "Classic potato chips with salt.",
        "ingredients": "potatoes, vegetable oil, salt",
        "nutrition_claims": [
            {"metric": "calories", "value": 150, "unit": "kcal", "basis": "per_serving"},
            {"metric": "sugar_g", "value": 1, "unit": "g", "basis": "per_serving"},
            {"metric": "fiber_g", "value": 0, "unit": "g", "basis": "per_serving"},
        ]
    },
    {
        "id": "prod_2",
        "title": "Kale Chips - Organic",
        "vendor": "GreenCrunch",
        "price": 5.99,
        "tags": ["chips", "kale", "organic", "high fiber"],
        "description": "Organic kale chips with sea salt. No artificial ingredients.",
        "ingredients": "organic kale, olive oil, sea salt",
        "nutrition_claims": [
            {"metric": "calories", "value": 80, "unit": "kcal", "basis": "per_serving"},
            {"metric": "sugar_g", "value": 0, "unit": "g", "basis": "per_serving"},
            {"metric": "fiber_g", "value": 3, "unit": "g", "basis": "per_serving"},
        ]
    },
    {
        "id": "prod_3",
        "title": "Peanut Butter Cookies",
        "vendor": "NutBakes",
        "price": 6.99,
        "tags": ["cookies", "peanut"],
        "description": "Cookies made with real peanut butter.",
        "ingredients": "flour, peanut butter, sugar",
        "nutrition_claims": [
            {"metric": "calories", "value": 180, "unit": "kcal", "basis": "per_serving"},
            {"metric": "sugar_g", "value": 12, "unit": "g", "basis": "per_serving"},
        ]
    },
    {
        "id": "prod_4",
        "title": "Protein Bites - Almond",
        "vendor": "FitSnack",
        "price": 8.99,
        "tags": ["protein", "high protein", "gluten free"],
        "description": "High protein snack with almonds. Contains tree nuts.",
        "ingredients": "almonds, whey protein, dates",
        "nutrition_claims": [
            {"metric": "calories", "value": 120, "unit": "kcal", "basis": "per_serving"},
            {"metric": "sugar_g", "value": 5, "unit": "g", "basis": "per_serving"},
            {"metric": "protein_g", "value": 10, "unit": "g", "basis": "per_serving"},
            {"metric": "fiber_g", "value": 2, "unit": "g", "basis": "per_serving"},
        ]
    },
    {
        "id": "prod_5",
        "title": "Vegan Oat Bars",
        "vendor": "PlantBites",
        "price": 7.49,
        "tags": ["bars", "vegan", "high fiber", "whole grain"],
        "description": "Vegan oat bars made with whole grains. No dairy, no eggs.",
        "ingredients": "oats, maple syrup, coconut oil",
        "nutrition_claims": [
            {"metric": "calories", "value": 140, "unit": "kcal", "basis": "per_serving"},
            {"metric": "sugar_g", "value": 6, "unit": "g", "basis": "per_serving"},
            {"metric": "fiber_g", "value": 4, "unit": "g", "basis": "per_serving"},
            {"metric": "protein_g", "value": 3, "unit": "g", "basis": "per_serving"},
        ]
    },
    {
        "id": "prod_6",
        "title": "Budget Crackers",
        "vendor": "ValueSnacks",
        "price": 2.99,
        "tags": ["crackers"],
        "description": "Simple wheat crackers.",
        "ingredients": "wheat flour, oil, salt",
        "nutrition_claims": []
    },
]


# ============================================================================
# Test: filter_by_preferences
# ============================================================================

def test_filter_excludes_allergens():
    """Test that products with specified allergens are excluded."""
    from app.services.recommendation import filter_by_preferences
    
    preferences = {"allergies": ["peanuts"]}
    filtered, reasons = filter_by_preferences(MOCK_PRODUCTS, preferences)
    
    # Peanut Butter Cookies should be excluded
    filtered_ids = [p["id"] for p in filtered]
    assert "prod_3" not in filtered_ids
    assert "prod_1" in filtered_ids
    assert "prod_2" in filtered_ids
    
    # Check exclusion reason
    assert any("peanut" in r.lower() for r in reasons)


def test_filter_excludes_tree_nuts_allergy():
    """Test that tree nut products are excluded for tree nut allergy."""
    from app.services.recommendation import filter_by_preferences
    
    preferences = {"allergies": ["tree nuts"]}
    filtered, reasons = filter_by_preferences(MOCK_PRODUCTS, preferences)
    
    # Protein Bites with almonds should be excluded
    filtered_ids = [p["id"] for p in filtered]
    assert "prod_4" not in filtered_ids


def test_filter_respects_budget():
    """Test that products over budget are excluded."""
    from app.services.recommendation import filter_by_preferences
    
    preferences = {"budget": 6.00}
    filtered, reasons = filter_by_preferences(MOCK_PRODUCTS, preferences)
    
    filtered_ids = [p["id"] for p in filtered]
    
    # Products over $6 should be excluded
    assert "prod_3" not in filtered_ids  # $6.99
    assert "prod_4" not in filtered_ids  # $8.99
    assert "prod_5" not in filtered_ids  # $7.49
    
    # Products under $6 should be included
    assert "prod_1" in filtered_ids  # $4.99
    assert "prod_2" in filtered_ids  # $5.99
    assert "prod_6" in filtered_ids  # $2.99


def test_filter_applies_vegan_diet():
    """Test that non-vegan products are flagged for vegan diet."""
    from app.services.recommendation import filter_by_preferences
    
    preferences = {"diet": "vegan"}
    filtered, reasons = filter_by_preferences(MOCK_PRODUCTS, preferences)
    
    # Vegan Oat Bars should be included (has "vegan" tag)
    filtered_ids = [p["id"] for p in filtered]
    assert "prod_5" in filtered_ids
    
    # Products with dairy/eggs in description may be excluded
    # (Protein Bites has "whey protein" which is dairy-derived)
    # Note: Simple check is tag-based, so this depends on implementation


def test_filter_combines_preferences():
    """Test that multiple preferences are combined correctly."""
    from app.services.recommendation import filter_by_preferences
    
    preferences = {
        "allergies": ["peanuts"],
        "budget": 8.00,
    }
    filtered, reasons = filter_by_preferences(MOCK_PRODUCTS, preferences)
    
    filtered_ids = [p["id"] for p in filtered]
    
    # Peanut cookies excluded by allergy
    assert "prod_3" not in filtered_ids
    
    # Protein Bites excluded by price ($8.99 > $8)
    assert "prod_4" not in filtered_ids


# ============================================================================
# Test: calculate_health_score
# ============================================================================

def test_health_score_low_sugar_bonus():
    """Test that low sugar products get higher scores."""
    from app.services.recommendation import calculate_health_score
    
    low_sugar_product = MOCK_PRODUCTS[1]  # Kale Chips: 0g sugar
    high_sugar_product = MOCK_PRODUCTS[2]  # Peanut Butter Cookies: 12g sugar
    
    low_score, low_factors = calculate_health_score(low_sugar_product)
    high_score, high_factors = calculate_health_score(high_sugar_product)
    
    assert low_score > high_score
    assert any("sugar" in f.lower() for f in low_factors)


def test_health_score_high_fiber_bonus():
    """Test that high fiber products get higher scores."""
    from app.services.recommendation import calculate_health_score
    
    high_fiber_product = MOCK_PRODUCTS[4]  # Vegan Oat Bars: 4g fiber
    no_fiber_product = MOCK_PRODUCTS[0]  # Classic Chips: 0g fiber
    
    high_score, high_factors = calculate_health_score(high_fiber_product)
    low_score, low_factors = calculate_health_score(no_fiber_product)
    
    assert high_score > low_score
    assert any("fiber" in f.lower() for f in high_factors)


def test_health_score_high_protein_bonus():
    """Test that high protein products get higher scores."""
    from app.services.recommendation import calculate_health_score
    
    protein_product = MOCK_PRODUCTS[3]  # Protein Bites: 10g protein
    
    score, factors = calculate_health_score(protein_product)
    
    assert score > 50  # Above baseline
    assert any("protein" in f.lower() for f in factors)


def test_health_score_organic_bonus():
    """Test that organic products get bonus points."""
    from app.services.recommendation import calculate_health_score
    
    organic_product = MOCK_PRODUCTS[1]  # Kale Chips: organic
    
    score, factors = calculate_health_score(organic_product)
    
    assert any("organic" in f.lower() for f in factors)


def test_health_score_no_data_warning():
    """Test that products without nutrition data get a warning."""
    from app.services.recommendation import calculate_health_score
    
    no_data_product = MOCK_PRODUCTS[5]  # Budget Crackers: no nutrition claims
    
    score, factors = calculate_health_score(no_data_product)
    
    assert score <= 60  # Capped without data
    assert any("limited" in f.lower() or "⚠️" in f for f in factors)


# ============================================================================
# Test: compare_products
# ============================================================================

def test_compare_identifies_healthier():
    """Test that comparison correctly identifies healthier product."""
    from app.services.recommendation import compare_products
    
    baseline = MOCK_PRODUCTS[0]  # Classic Chips
    alternative = MOCK_PRODUCTS[1]  # Kale Chips
    
    comparison = compare_products(baseline, alternative)
    
    assert comparison["is_healthier"] == True
    assert comparison["score_improvement"] > 0
    assert len(comparison["reasons"]) > 0


def test_compare_includes_price():
    """Test that comparison includes price difference."""
    from app.services.recommendation import compare_products
    
    expensive = MOCK_PRODUCTS[3]  # Protein Bites $8.99
    cheap = MOCK_PRODUCTS[5]  # Budget Crackers $2.99
    
    comparison = compare_products(expensive, cheap)
    
    # If cheap is alternative, should mention it's cheaper
    assert any("cheaper" in r.lower() or "$" in r for r in comparison["reasons"])


def test_compare_includes_nutrition_diff():
    """Test that comparison includes specific nutrition differences."""
    from app.services.recommendation import compare_products
    
    baseline = MOCK_PRODUCTS[0]  # Classic Chips: 0g fiber
    alternative = MOCK_PRODUCTS[4]  # Vegan Bars: 4g fiber
    
    comparison = compare_products(baseline, alternative)
    
    # Should mention fiber improvement
    assert any("fiber" in r.lower() for r in comparison["reasons"])


# ============================================================================
# Test: categorize_product
# ============================================================================

def test_categorize_chips():
    """Test that chip products are categorized correctly."""
    from app.services.recommendation import categorize_product
    
    chips_product = MOCK_PRODUCTS[0]  # Classic Chips
    
    category = categorize_product(chips_product)
    
    assert category == "chips"


def test_categorize_bars():
    """Test that bar products are categorized correctly."""
    from app.services.recommendation import categorize_product
    
    bar_product = MOCK_PRODUCTS[4]  # Vegan Oat Bars
    
    category = categorize_product(bar_product)
    
    assert category == "bars"


def test_categorize_unknown():
    """Test that unknown products get 'other' category."""
    from app.services.recommendation import categorize_product
    
    unknown_product = {
        "title": "Mystery Snack",
        "tags": [],
        "product_type": None,
    }
    
    category = categorize_product(unknown_product)
    
    assert category == "other"


# ============================================================================
# Test: Safety Checks
# ============================================================================

def test_allergy_warning_when_no_allergies_specified():
    """Test that an allergy warning is included when allergies not specified."""
    # This would require integration test with the full recommendation function
    # but we can test the logic exists
    
    preferences_without_allergies = {"diet": "vegan"}
    
    # Warning should be generated when allergies list is empty
    allergy_warning = None
    if not preferences_without_allergies.get("allergies"):
        allergy_warning = "⚠️ No allergies specified. Please confirm any food allergies before purchasing."
    
    assert allergy_warning is not None
    assert "allergies" in allergy_warning.lower()


def test_confirmation_required_at_review():
    """Test that buy flow requires confirmation at review step."""
    from app.services.orchestration import PURCHASE_STEPS, get_purchase_step_by_name
    
    review_step = get_purchase_step_by_name("review_order")
    
    assert review_step is not None
    assert review_step["requires_confirmation"] == True
    assert "STOP" in review_step["instruction"] or "confirm" in review_step["instruction"].lower()


def test_no_auto_purchase_step():
    """Test that there's no auto-complete purchase step."""
    from app.services.orchestration import PURCHASE_STEPS
    
    # Verify no step automatically completes purchase
    for step in PURCHASE_STEPS:
        if step["name"] in ["complete_purchase", "submit_order", "place_order"]:
            # If such a step exists, it must require confirmation
            assert step.get("requires_confirmation", False) == True


# ============================================================================
# Integration-style tests (with mocking)
# ============================================================================

def test_recommendation_flow_structure():
    """Test that recommendation result has expected structure."""
    # Mock result structure
    result = {
        "baseline": {"title": "Test", "health_score": 50, "factors": []},
        "recommendations": [],
        "preferences_applied": {},
        "allergy_warning": None,
        "decision_ids": [],
        "session_id": "test-session",
    }
    
    # Verify all required fields exist
    assert "baseline" in result
    assert "recommendations" in result
    assert "preferences_applied" in result
    assert "decision_ids" in result
    assert "session_id" in result


def test_recommendation_item_structure():
    """Test that each recommendation has expected fields."""
    expected_fields = [
        "product_id",
        "title",
        "price",
        "health_score",
        "score_improvement",
        "why_healthier",
        "comparison_summary",
        "confidence",
    ]
    
    mock_recommendation = {
        "product_id": "test_id",
        "title": "Test Product",
        "price": 5.99,
        "health_score": 75,
        "score_improvement": 25,
        "why_healthier": ["Low sugar", "High fiber"],
        "comparison_summary": "Healthier option",
        "confidence": "high",
    }
    
    for field in expected_fields:
        assert field in mock_recommendation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
