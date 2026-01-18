"""
Shopify Catalog Ingestion Service

Ingests products from a Shopify store into the Neo4j graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests

from app.config import settings
from app.services.graph import graph_service
from app.services.doc_ingestion import embed_text

logger = logging.getLogger(__name__)


# Common nutrition keywords for parsing
NUTRITION_PATTERNS = {
    "calories": r"(\d+)\s*(?:cal|calories|kcal)",
    "sugar_g": r"(\d+(?:\.\d+)?)\s*g?\s*(?:sugar|sugars)",
    "fiber_g": r"(\d+(?:\.\d+)?)\s*g?\s*(?:fiber|fibre|dietary fiber)",
    "protein_g": r"(\d+(?:\.\d+)?)\s*g?\s*(?:protein)",
    "fat_g": r"(\d+(?:\.\d+)?)\s*g?\s*(?:fat|total fat)",
    "sodium_mg": r"(\d+)\s*mg?\s*(?:sodium|salt)",
    "carbs_g": r"(\d+(?:\.\d+)?)\s*g?\s*(?:carb|carbs|carbohydrate|carbohydrates)",
}


def parse_nutrition_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse nutrition facts from product description or metafields.
    
    Returns list of {metric, value, unit, basis}
    """
    claims = []
    text_lower = text.lower()
    
    for metric, pattern in NUTRITION_PATTERNS.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                unit = "g" if metric.endswith("_g") else ("mg" if metric.endswith("_mg") else "kcal")
                claims.append({
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "basis": "per_serving"
                })
            except ValueError:
                continue
    
    return claims


def parse_ingredients(description: str) -> Optional[str]:
    """Extract ingredients list from description."""
    # Look for ingredients section
    patterns = [
        r"ingredients?[:\s]+([^.]+(?:\.|$))",
        r"contains?[:\s]+([^.]+(?:\.|$))",
        r"made with[:\s]+([^.]+(?:\.|$))",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


async def fetch_shopify_products(
    store_url: str,
    access_token: Optional[str] = None,
    limit: int = 250
) -> List[Dict[str, Any]]:
    """
    Fetch products from a Shopify store.
    
    Uses the public products.json endpoint if no access token,
    or the Admin API if access token is provided.
    """
    products = []
    
    # Clean store URL
    store_url = store_url.rstrip("/")
    if not store_url.startswith("http"):
        store_url = f"https://{store_url}"
    
    if access_token:
        # Use Admin API
        headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json"
        }
        url = f"{store_url}/admin/api/2024-01/products.json?limit={limit}"
    else:
        # Use public products.json (limited data)
        headers = {}
        url = f"{store_url}/products.json?limit={limit}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        products = data.get("products", [])
        logger.info(f"Fetched {len(products)} products from {store_url}")
    except Exception as e:
        logger.error(f"Failed to fetch products from {store_url}: {e}")
        raise
    
    return products


def transform_shopify_product(
    product: Dict[str, Any],
    store_url: str
) -> Dict[str, Any]:
    """Transform a Shopify product into our product format."""
    # Get first variant for pricing
    variants = product.get("variants", [])
    price = None
    if variants:
        try:
            price = float(variants[0].get("price", 0))
        except (ValueError, TypeError):
            price = None
    
    # Get first image
    images = product.get("images", [])
    image_url = images[0].get("src") if images else None
    
    # Extract description text
    description = product.get("body_html", "") or ""
    # Strip HTML tags
    description = re.sub(r"<[^>]+>", " ", description)
    description = re.sub(r"\s+", " ", description).strip()
    
    # Build product URL
    handle = product.get("handle", "")
    store_url = store_url.rstrip("/")
    url = f"{store_url}/products/{handle}" if handle else None
    
    return {
        "shopify_id": str(product.get("id", "")),
        "title": product.get("title", ""),
        "handle": handle,
        "vendor": product.get("vendor"),
        "product_type": product.get("product_type"),
        "tags": product.get("tags", "").split(", ") if isinstance(product.get("tags"), str) else product.get("tags", []),
        "description": description[:2000],  # Limit description length
        "price": price,
        "url": url,
        "image_url": image_url,
    }


async def ingest_shopify_catalog(
    company_id: str,
    store_url: str,
    access_token: Optional[str] = None,
    max_products: int = 250
) -> Dict[str, Any]:
    """
    Ingest a Shopify store's catalog into the graph.
    
    Returns summary of ingested products.
    """
    logger.info(f"Starting Shopify catalog ingestion for company {company_id}")
    
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise ValueError(f"Company {company_id} not found")
    
    # Fetch products from Shopify
    raw_products = await fetch_shopify_products(store_url, access_token, max_products)
    
    products_created = 0
    nutrition_claims_created = 0
    
    for raw_product in raw_products:
        try:
            # Transform product data
            product_data = transform_shopify_product(raw_product, store_url)
            
            # Generate embedding for product
            embed_text_content = f"{product_data['title']} {product_data.get('description', '')} {' '.join(product_data.get('tags', []))}"
            embedding = embed_text(embed_text_content[:1000])  # Limit embedding text
            
            # Parse ingredients from description
            ingredients = parse_ingredients(product_data.get("description", ""))
            
            # Create product node
            product = graph_service.create_product(
                company_id=company_id,
                title=product_data["title"],
                handle=product_data["handle"],
                vendor=product_data.get("vendor"),
                price=product_data.get("price"),
                currency="CAD",
                product_type=product_data.get("product_type"),
                tags=product_data.get("tags"),
                description=product_data.get("description"),
                ingredients=ingredients,
                url=product_data.get("url"),
                image_url=product_data.get("image_url"),
                embedding=embedding,
                shopify_id=product_data.get("shopify_id")
            )
            
            if product:
                products_created += 1
                
                # Parse and add nutrition claims from description
                description = product_data.get("description", "")
                nutrition_claims = parse_nutrition_from_text(description)
                
                for claim in nutrition_claims:
                    graph_service.add_nutrition_claim(
                        product_id=product["id"],
                        metric=claim["metric"],
                        value=claim["value"],
                        unit=claim["unit"],
                        basis=claim["basis"]
                    )
                    nutrition_claims_created += 1
        
        except Exception as e:
            logger.warning(f"Failed to ingest product {raw_product.get('title', 'unknown')}: {e}")
            continue
    
    result = {
        "company_id": company_id,
        "store_url": store_url,
        "products_fetched": len(raw_products),
        "products_created": products_created,
        "nutrition_claims_created": nutrition_claims_created,
        "status": "completed"
    }
    
    logger.info(f"Shopify catalog ingestion complete: {result}")
    return result


async def sync_product_from_url(
    company_id: str,
    product_url: str
) -> Optional[Dict[str, Any]]:
    """
    Sync a single product from its URL.
    
    Useful for adding products on-demand during recommendation.
    """
    try:
        # Parse store URL and handle from product URL
        # Format: https://store.myshopify.com/products/product-handle
        match = re.match(r"(https?://[^/]+)/products/([^/?]+)", product_url)
        if not match:
            logger.warning(f"Invalid product URL format: {product_url}")
            return None
        
        store_url = match.group(1)
        handle = match.group(2)
        
        # Fetch product data
        product_json_url = f"{store_url}/products/{handle}.json"
        response = requests.get(product_json_url, timeout=10)
        response.raise_for_status()
        
        product_data = response.json().get("product", {})
        transformed = transform_shopify_product(product_data, store_url)
        
        # Generate embedding
        embed_text_content = f"{transformed['title']} {transformed.get('description', '')} {' '.join(transformed.get('tags', []))}"
        embedding = embed_text(embed_text_content[:1000])
        
        # Create product
        product = graph_service.create_product(
            company_id=company_id,
            title=transformed["title"],
            handle=transformed["handle"],
            vendor=transformed.get("vendor"),
            price=transformed.get("price"),
            currency="CAD",
            product_type=transformed.get("product_type"),
            tags=transformed.get("tags"),
            description=transformed.get("description"),
            ingredients=parse_ingredients(transformed.get("description", "")),
            url=product_url,
            image_url=transformed.get("image_url"),
            embedding=embedding,
            shopify_id=transformed.get("shopify_id")
        )
        
        return product
    
    except Exception as e:
        logger.error(f"Failed to sync product from {product_url}: {e}")
        return None
