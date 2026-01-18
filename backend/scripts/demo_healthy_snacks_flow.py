#!/usr/bin/env python3
"""
Demo: Healthy Snack Alternatives Flow

This script demonstrates the full flow:
1. Ingest Shopify catalog for a demo store
2. Create a user profile with allergy + budget preferences
3. Request healthier alternatives to a baseline snack
4. Print recommended products with justification
5. Simulate selecting one and running the buy flow
6. Print Decision trace IDs

Usage:
    cd backend
    source venv/bin/activate
    python scripts/demo_healthy_snacks_flow.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.graph import graph_service
from app.services.shopify_catalog import ingest_shopify_catalog
from app.services.recommendation import recommend_healthy_alternatives
from app.services.orchestration import plan_buy_action


async def demo_flow():
    """Run the full demo flow."""
    print("=" * 60)
    print("HEALTHY SNACK ALTERNATIVES - DEMO FLOW")
    print("=" * 60)
    
    # Step 0: Verify Neo4j connection
    print("\n[Step 0] Verifying Neo4j connection...")
    if not graph_service.verify_connectivity():
        print("❌ Neo4j is not available. Please start Neo4j first.")
        print("   Run: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5")
        return
    print("✅ Neo4j connected")
    
    # Step 1: Setup schema
    print("\n[Step 1] Setting up Neo4j schema...")
    graph_service.setup_schema()
    graph_service.setup_vector_index()
    print("✅ Schema setup complete")
    
    # Step 2: Create a demo company
    print("\n[Step 2] Creating demo company...")
    company = graph_service.create_company(
        name="Demo Healthy Snacks Store",
        domain="demo-snacks.myshopify.com"
    )
    company_id = company["id"]
    print(f"✅ Created company: {company['name']} (ID: {company_id})")
    
    # Step 3: Create demo products (simulating Shopify catalog)
    print("\n[Step 3] Creating demo product catalog...")
    
    # Real PUBLIC Shopify stores with healthy snacks
    # These are actual Shopify storefronts that sell healthy products
    demo_products = [
        {
            "title": "Classic Potato Chips - Original",
            "handle": "classic-potato-chips",
            "vendor": "ChipCo",
            "price": 4.99,
            "product_type": "chips",
            "tags": ["chips", "salty", "snack"],
            "description": "Classic potato chips. Ingredients: potatoes, vegetable oil, salt. Nutrition: 150 calories, 10g fat, 15g carbs, 1g protein, 0g fiber, 170mg sodium, 1g sugar per serving.",
            "store_url": "https://lairdsuperfood.com/collections/all",  # Laird Superfood - Shopify
        },
        {
            "title": "Organic Banana Bites",
            "handle": "organic-banana-bites",
            "vendor": "Barnana",
            "price": 5.99,
            "product_type": "chips",
            "tags": ["chips", "banana", "healthy", "organic", "high fiber"],
            "description": "Organic banana snacks. No artificial ingredients. Nutrition: 80 calories, 4g fat, 8g carbs, 3g protein, 2g fiber, 120mg sodium, 0g sugar per serving. Made with whole ingredients.",
            "store_url": "https://barnana.com/collections/all-snacks",  # Barnana - Shopify
        },
        {
            "title": "Perfect Bar - Peanut Butter",
            "handle": "perfect-bar-peanut-butter",
            "vendor": "Perfect Snacks",
            "price": 4.49,
            "product_type": "bars",
            "tags": ["bars", "protein", "peanut butter", "refrigerated"],
            "description": "Refrigerated protein bar with peanut butter. Nutrition: 330 calories, 17g fat, 27g carbs, 17g protein, 3g fiber, 150mg sodium, 19g sugar per serving.",
            "store_url": "https://perfectsnacks.com/collections/all",  # Perfect Snacks - Shopify
        },
        {
            "title": "Superfood Creamer",
            "handle": "superfood-creamer",
            "vendor": "Laird Superfood",
            "price": 11.99,
            "product_type": "superfood",
            "tags": ["superfood", "creamer", "coconut", "vegan", "keto"],
            "description": "Plant-based superfood creamer. High in MCTs, vegan, keto-friendly. Nutrition: 45 calories, 4g fat, 1g carbs, 0g protein, 0g fiber, 0mg sodium, 0g sugar per serving. Gluten free and vegan.",
            "store_url": "https://lairdsuperfood.com/collections/creamers",  # Laird Superfood - Shopify
        },
        {
            "title": "Hu Chocolate Bar",
            "handle": "hu-chocolate-bar",
            "vendor": "Hu Kitchen",
            "price": 6.99,
            "product_type": "chocolate",
            "tags": ["chocolate", "vegan", "paleo", "no refined sugar"],
            "description": "Vegan, paleo-friendly dark chocolate bar. No refined sugar, dairy-free. Nutrition: 200 calories, 15g fat, 18g carbs, 2g protein, 3g fiber, 0mg sodium, 8g sugar per serving.",
            "store_url": "https://hukitchen.com/collections/chocolate-bars",  # Hu Kitchen - Shopify
        },
        {
            "title": "Coconut Chips - Cacao",
            "handle": "coconut-chips-cacao",
            "vendor": "Dang Foods",
            "price": 5.49,
            "product_type": "chips",
            "tags": ["coconut", "chips", "keto", "low sugar", "high fiber", "vegan"],
            "description": "Crispy coconut chips with cacao. Keto-friendly, low sugar snack. Nutrition: 140 calories, 10g fat, 12g carbs, 2g protein, 4g fiber, 80mg sodium, 4g sugar per serving.",
            "store_url": "https://dangfoods.com/collections/all",  # Dang Foods - Shopify
        },
        {
            "title": "Protein Cookie - Chocolate Chip",
            "handle": "protein-cookie-chocolate",
            "vendor": "Lenny & Larry's",
            "price": 3.49,
            "product_type": "cookies",
            "tags": ["protein", "cookie", "vegan", "high protein"],
            "description": "Vegan protein cookie with 16g protein. Plant-based, no dairy. Nutrition: 400 calories, 16g fat, 50g carbs, 16g protein, 6g fiber, 400mg sodium, 26g sugar per serving.",
            "store_url": "https://www.lennylarry.com/collections/all",  # Lenny & Larry's - Shopify
        },
    ]
    
    created_products = []
    for p in demo_products:
        # Create product with nutrition parsing
        from app.services.shopify_catalog import parse_nutrition_from_text
        from app.services.doc_ingestion import embed_text
        
        # Generate embedding
        embed_text_content = f"{p['title']} {p.get('description', '')} {' '.join(p.get('tags', []))}"
        embedding = embed_text(embed_text_content[:1000])
        
        product = graph_service.create_product(
            company_id=company_id,
            title=p["title"],
            handle=p["handle"],
            vendor=p["vendor"],
            price=p["price"],
            currency="CAD",
            product_type=p["product_type"],
            tags=p["tags"],
            description=p["description"],
            url=p.get("store_url", f"https://www.amazon.ca/s?k={p['title'].replace(' ', '+')}"),
            embedding=embedding
        )
        
        if product:
            created_products.append(product)
            
            # Parse and add nutrition claims
            nutrition_claims = parse_nutrition_from_text(p["description"])
            for claim in nutrition_claims:
                graph_service.add_nutrition_claim(
                    product_id=product["id"],
                    metric=claim["metric"],
                    value=claim["value"],
                    unit=claim["unit"],
                    basis=claim["basis"]
                )
            
            print(f"   ✅ Created: {p['title']} - ${p['price']} ({len(nutrition_claims)} nutrition claims)")
    
    print(f"✅ Created {len(created_products)} products")
    
    # Step 4: Create user profile with preferences
    print("\n[Step 4] Creating user profile with preferences...")
    
    user_id = "demo_user_123"
    profile = graph_service.create_user_profile(
        user_id=user_id,
        age_group="adult"
    )
    
    # Add preferences
    graph_service.add_user_preference(user_id, "allergy", "peanuts")
    graph_service.add_user_preference(user_id, "budget", "8 CAD")
    graph_service.add_user_preference(user_id, "diet", "vegetarian")
    
    profile = graph_service.get_user_profile(user_id)
    print(f"✅ Created profile for user: {user_id}")
    prefs_str = [f"{p['type']}: {p['value']}" for p in profile['preferences']]
    print(f"   Preferences: {prefs_str}")
    
    # Step 5: Request healthy alternatives to potato chips
    print("\n[Step 5] Requesting healthy alternatives to 'Classic Potato Chips'...")
    
    baseline_product = created_products[0]  # Classic Potato Chips
    
    result = await recommend_healthy_alternatives(
        company_id=company_id,
        baseline_product_id=baseline_product["id"],
        user_id=user_id,
        preferences={
            "allergies": ["peanuts"],
            "budget": 8.0,
        },
        limit=3
    )
    
    print(f"\n📊 BASELINE: {result['baseline']['title']}")
    print(f"   Health Score: {result['baseline']['health_score']:.0f}/100")
    print(f"   Factors: {', '.join(result['baseline']['factors'][:3])}")
    
    print(f"\n🥗 RECOMMENDATIONS ({len(result['recommendations'])} found):")
    print("-" * 50)
    
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   💰 Price: ${rec['price']} {rec['currency']}")
        print(f"   📈 Health Score: {rec['health_score']:.0f}/100 (+{rec['score_improvement']:.0f} improvement)")
        print(f"   🏷️  Confidence: {rec['confidence']}")
        print(f"   ✅ Why healthier: {', '.join(rec['why_healthier'][:3])}")
        print(f"   📝 Comparison: {rec['comparison_summary']}")
    
    if result.get("allergy_warning"):
        print(f"\n⚠️  {result['allergy_warning']}")
    
    print(f"\n📋 Decision IDs: {result['decision_ids']}")
    print(f"   Session ID: {result['session_id']}")
    
    # Step 6: Simulate buy flow for first recommendation
    if result["recommendations"]:
        print("\n" + "=" * 60)
        print("[Step 6] Simulating buy flow for first recommendation...")
        print("=" * 60)
        
        selected_product_id = result["recommendations"][0]["product_id"]
        selected_title = result["recommendations"][0]["title"]
        
        print(f"\n🛒 Selected product: {selected_title}")
        
        # Simulate UI context at different stages
        session_id = result["session_id"]
        
        # Create cart session
        cart = graph_service.create_cart_session(user_id, session_id)
        
        ui_contexts = [
            {
                "step": "Starting from homepage",
                "context": {
                    "url": "https://demo-snacks.myshopify.com/",
                    "title": "Demo Healthy Snacks Store",
                    "elements": [
                        {"selector": "input[type='search']", "text": "", "type": "input"},
                        {"selector": ".cart-icon", "text": "Cart", "type": "button"},
                    ]
                }
            },
            {
                "step": "On product page",
                "context": {
                    "url": f"https://demo-snacks.myshopify.com/products/{result['recommendations'][0]['title'].lower().replace(' ', '-')}",
                    "title": selected_title,
                    "elements": [
                        {"selector": "button[name='add']", "text": "Add to Cart", "type": "button"},
                        {"selector": ".cart-icon", "text": "Cart (0)", "type": "button"},
                    ]
                }
            },
            {
                "step": "Product added to cart",
                "context": {
                    "url": f"https://demo-snacks.myshopify.com/products/{result['recommendations'][0]['title'].lower().replace(' ', '-')}",
                    "title": selected_title,
                    "visible_text": "Added to your cart!",
                    "elements": [
                        {"selector": ".cart-icon", "text": "Cart (1)", "type": "button"},
                        {"selector": "a[href='/cart']", "text": "View Cart", "type": "link"},
                    ]
                }
            },
            {
                "step": "On cart page",
                "context": {
                    "url": "https://demo-snacks.myshopify.com/cart",
                    "title": "Your Cart",
                    "elements": [
                        {"selector": "button[name='checkout']", "text": "Proceed to Checkout", "type": "button"},
                    ]
                }
            },
            {
                "step": "On checkout",
                "context": {
                    "url": "https://demo-snacks.myshopify.com/checkout/payment",
                    "title": "Checkout - Payment",
                    "elements": [
                        {"selector": ".review-order", "text": "Review Order", "type": "button"},
                    ]
                }
            },
        ]
        
        decision_ids = []
        
        for stage in ui_contexts:
            print(f"\n📍 {stage['step']}")
            
            buy_result = await plan_buy_action(
                company_id=company_id,
                product_id=selected_product_id,
                ui_context=stage["context"],
                session_state={
                    "session_id": session_id,
                    "cart_id": session_id,
                }
            )
            
            print(f"   Next step: {buy_result.get('step_name', 'N/A')}")
            
            if buy_result.get("action"):
                action = buy_result["action"]
                print(f"   Action: {action.get('type', 'N/A')} - {action.get('instruction', 'N/A')[:50]}")
            
            if buy_result.get("decision_id"):
                decision_ids.append(buy_result["decision_id"])
            
            # Check for confirmation required
            if buy_result.get("requires_confirmation"):
                print(f"\n⚠️  STOP: {buy_result['action']['instruction']}")
                print(f"   Confirmation required before proceeding!")
                print(f"   Token format: CONFIRM_{session_id}_{selected_product_id}")
                break
        
        print(f"\n📋 Buy Flow Decision IDs: {decision_ids}")
    
    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"""
✅ Created company: {company['name']}
✅ Ingested {len(created_products)} products with nutrition data
✅ Created user profile with allergy/budget preferences
✅ Generated {len(result['recommendations'])} healthier recommendations
✅ Recorded decision traces for full justification
✅ Buy flow stops at review step (requires confirmation)

Key Points:
- Recommendations focus on lower sugar, higher fiber/protein
- All excluded products due to preferences are logged
- Decision traces link to evidence and nutrition claims
- No auto-purchase: explicit confirmation required

Neo4j Cypher to explore:
  MATCH (d:Decision) RETURN d LIMIT 5
  MATCH (p:Product)-[:HAS_NUTRITION]->(n:NutritionClaim) RETURN p.title, n
  MATCH (c:Comparison)-[:BASELINE]->(b:Product), (c)-[:ALTERNATIVE]->(a:Product) RETURN b.title, a.title, c.reason_summary
""")


if __name__ == "__main__":
    asyncio.run(demo_flow())
