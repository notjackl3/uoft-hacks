#!/usr/bin/env python3
"""
Ingest Shopify Catalog Script

Ingests products from a Shopify store into the Neo4j graph.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/ingest_shopify_catalog.py --store-url https://example.myshopify.com --company-id <id>

Or for a demo with a new company:
    python scripts/ingest_shopify_catalog.py --store-url https://example.myshopify.com --create-company "My Store"
"""

import argparse
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    parser = argparse.ArgumentParser(description="Ingest Shopify catalog into Neo4j")
    parser.add_argument("--store-url", required=True, help="Shopify store URL (e.g., https://example.myshopify.com)")
    parser.add_argument("--company-id", help="Existing company ID to add products to")
    parser.add_argument("--create-company", help="Create a new company with this name")
    parser.add_argument("--access-token", help="Shopify Admin API access token (optional)")
    parser.add_argument("--max-products", type=int, default=250, help="Maximum products to ingest")
    
    args = parser.parse_args()
    
    # Import services
    from app.services.graph import graph_service
    from app.services.shopify_catalog import ingest_shopify_catalog
    
    # Verify Neo4j connection
    print("Checking Neo4j connection...")
    if not graph_service.verify_connectivity():
        print("❌ Cannot connect to Neo4j. Please ensure it's running.")
        print("   Start with: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5")
        sys.exit(1)
    print("✅ Neo4j connected")
    
    # Setup schema
    print("Setting up schema...")
    graph_service.setup_schema()
    graph_service.setup_vector_index()
    
    # Get or create company
    company_id = args.company_id
    
    if args.create_company:
        print(f"Creating company: {args.create_company}")
        company = graph_service.create_company(
            name=args.create_company,
            domain=args.store_url.replace("https://", "").replace("http://", "")
        )
        company_id = company["id"]
        print(f"✅ Created company with ID: {company_id}")
    
    if not company_id:
        print("❌ Must provide --company-id or --create-company")
        sys.exit(1)
    
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        print(f"❌ Company {company_id} not found")
        sys.exit(1)
    
    print(f"Ingesting products for company: {company['name']}")
    
    # Ingest catalog
    result = await ingest_shopify_catalog(
        company_id=company_id,
        store_url=args.store_url,
        access_token=args.access_token,
        max_products=args.max_products
    )
    
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"Store URL: {result['store_url']}")
    print(f"Products fetched: {result['products_fetched']}")
    print(f"Products created: {result['products_created']}")
    print(f"Nutrition claims: {result['nutrition_claims_created']}")
    print(f"Status: {result['status']}")
    
    print(f"\nCompany ID for API calls: {company_id}")
    print("\nExample API calls:")
    print(f"  GET  /api/commerce/{company_id}/products")
    print(f"  POST /api/commerce/{company_id}/recommendations/snacks")


if __name__ == "__main__":
    asyncio.run(main())
