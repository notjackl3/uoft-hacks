"""
Web Search Service for Product Evidence

Searches the web for healthier alternatives and extracts evidence snippets.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.services.graph import graph_service

logger = logging.getLogger(__name__)


# Trusted health/nutrition sources for higher evidence quality
TRUSTED_SOURCES = [
    "healthline.com",
    "medicalnewstoday.com",
    "eatthis.com",
    "prevention.com",
    "self.com",
    "health.com",
    "webmd.com",
    "mayoclinic.org",
    "nutritionix.com",
    "myfitnesspal.com",
    "foodnetwork.com",
    "allrecipes.com",
]


async def search_web(
    query: str,
    num_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search the web using a simple scraping approach.
    
    Note: In production, use a proper search API (Google Custom Search, Bing, etc.)
    
    Returns list of {url, title, snippet}
    """
    results = []
    
    # Use DuckDuckGo HTML for simple scraping (no API key needed)
    # Note: This is for demo purposes. Production should use a proper API.
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; HealthySnackBot/1.0)"
        }
        
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for result in soup.select(".result")[:num_results]:
            title_elem = result.select_one(".result__title")
            snippet_elem = result.select_one(".result__snippet")
            link_elem = result.select_one(".result__url")
            
            if title_elem and snippet_elem:
                # Extract URL from the result
                url = ""
                if link_elem:
                    url = link_elem.get_text(strip=True)
                    if not url.startswith("http"):
                        url = f"https://{url}"
                
                results.append({
                    "url": url,
                    "title": title_elem.get_text(strip=True),
                    "snippet": snippet_elem.get_text(strip=True)[:500]
                })
        
        logger.info(f"Web search for '{query[:50]}...' returned {len(results)} results")
    
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
    
    return results


async def search_healthier_alternatives(
    baseline_product: str,
    additional_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for healthier alternatives to a snack product.
    
    Returns list of evidence snippets with sources.
    """
    queries = [
        f"{baseline_product} healthier alternatives",
        f"{baseline_product} low sugar high fiber alternatives",
        f"healthy snacks similar to {baseline_product}",
    ]
    
    if additional_context:
        queries.append(f"{baseline_product} {additional_context} alternatives")
    
    all_evidence = []
    seen_urls = set()
    
    for query in queries[:2]:  # Limit to avoid rate limiting
        results = await search_web(query, num_results=5)
        
        for result in results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Determine source quality
            is_trusted = any(source in url.lower() for source in TRUSTED_SOURCES)
            
            all_evidence.append({
                "source_type": "web_page",
                "source_ref": url,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "query": query,
                "is_trusted": is_trusted,
                "fetched_at": datetime.utcnow().isoformat()
            })
    
    # Sort by trusted sources first
    all_evidence.sort(key=lambda x: (not x["is_trusted"], x["source_ref"]))
    
    return all_evidence


async def fetch_page_content(url: str, max_length: int = 5000) -> Optional[str]:
    """Fetch and extract main content from a web page."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; HealthySnackBot/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()
        
        # Get main content
        main = soup.find("main") or soup.find("article") or soup.body
        if main:
            text = main.get_text(separator="\n", strip=True)
            return text[:max_length]
        
        return None
    
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def extract_product_mentions(
    text: str,
    catalog_products: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract mentions of catalog products from text.
    
    Returns list of matched products with relevance context.
    """
    matches = []
    text_lower = text.lower()
    
    for product in catalog_products:
        title_lower = product.get("title", "").lower()
        
        # Simple title matching
        if title_lower in text_lower:
            matches.append({
                "product_id": product["id"],
                "title": product["title"],
                "match_type": "exact_title"
            })
            continue
        
        # Brand/vendor matching
        vendor = product.get("vendor", "").lower()
        if vendor and len(vendor) > 2 and vendor in text_lower:
            matches.append({
                "product_id": product["id"],
                "title": product["title"],
                "match_type": "vendor"
            })
            continue
        
        # Tag/category matching
        tags = product.get("tags", [])
        for tag in tags:
            if tag.lower() in text_lower:
                matches.append({
                    "product_id": product["id"],
                    "title": product["title"],
                    "match_type": "tag",
                    "matched_tag": tag
                })
                break
    
    return matches


async def gather_product_evidence(
    product_id: str,
    product_title: str,
    company_id: str
) -> List[Dict[str, Any]]:
    """
    Gather external evidence for a specific product.
    
    Stores evidence in the graph and returns evidence IDs.
    """
    evidence_ids = []
    
    # Search for product information
    queries = [
        f"{product_title} nutrition facts",
        f"{product_title} ingredients healthy",
        f"{product_title} review health",
    ]
    
    for query in queries[:2]:
        results = await search_web(query, num_results=3)
        
        for result in results:
            if not result.get("snippet"):
                continue
            
            # Store evidence in graph
            evidence = graph_service.add_product_evidence(
                product_id=product_id,
                source_type="web_page",
                source_ref=result.get("url", ""),
                snippet=result.get("snippet", "")[:500],
                fetched_at=datetime.utcnow()
            )
            
            if evidence:
                evidence_ids.append(evidence["id"])
    
    return evidence_ids


def parse_health_claims_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse health-related claims from text.
    
    Returns list of health claims with their sentiment (positive/negative).
    """
    claims = []
    text_lower = text.lower()
    
    # Positive health indicators
    positive_patterns = [
        (r"low[\s-]?sugar", "low_sugar", "positive"),
        (r"no[\s-]?added[\s-]?sugar", "no_added_sugar", "positive"),
        (r"high[\s-]?fiber", "high_fiber", "positive"),
        (r"high[\s-]?protein", "high_protein", "positive"),
        (r"whole[\s-]?grain", "whole_grain", "positive"),
        (r"organic", "organic", "positive"),
        (r"natural", "natural", "positive"),
        (r"no[\s-]?artificial", "no_artificial", "positive"),
        (r"gluten[\s-]?free", "gluten_free", "neutral"),
        (r"vegan", "vegan", "neutral"),
        (r"plant[\s-]?based", "plant_based", "positive"),
    ]
    
    # Negative health indicators
    negative_patterns = [
        (r"high[\s-]?sugar", "high_sugar", "negative"),
        (r"processed", "processed", "negative"),
        (r"artificial", "artificial", "negative"),
        (r"preservatives", "preservatives", "negative"),
    ]
    
    for pattern, claim_type, sentiment in positive_patterns + negative_patterns:
        if re.search(pattern, text_lower):
            claims.append({
                "type": claim_type,
                "sentiment": sentiment
            })
    
    return claims
