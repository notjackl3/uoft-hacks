# File: scripts/seed_cache.py

import asyncio
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb+srv://Phin:Nmk8GE0hIRZcpcOG@big-bro.mrugz9b.mongodb.net/?appName=Big-Bro"  # Update with your connection string
DB_NAME = "big_bro"  # Update with your database name


async def seed_facebook_tennis_cache():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db.plan_cache
    
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=30)
    
    cache_doc = {
        "cache_id": "manual-facebook-tennis-group-001",
        
        # Goal matching fields
        "canonical_goal": "go on facebook search for tennis group and join it",
        "original_user_goal": "i want a guide to go on facebook, search for a tennis group, and join it",
        "goal_keywords": ["facebook", "search", "tennis", "group", "join"],
        "goal_embedding": [],  # Can populate with embed_text() if needed
        
        # Context
        "target_domain": "facebook",
        "target_url": "https://www.facebook.com",
        "page_context": "homepage",
        
        # The successful plan
        "planned_steps": [
            {
                "step_number": 1,
                "action": "WAIT",
                "description": "Navigate to Facebook if not already there. Go to https://www.facebook.com",
                "text_input": None,
                "target_hints": {
                    "type": None,
                    "text_contains": [],
                    "placeholder_contains": [],
                    "role": None,
                    "selector_pattern": None
                },
                "expected_page_change": True
            },
            {
                "step_number": 2,
                "action": "CLICK",
                "description": "Click on the search bar at the top of Facebook",
                "text_input": None,
                "target_hints": {
                    "type": "input",
                    "text_contains": ["search", "facebook"],
                    "placeholder_contains": ["search"],
                    "role": "searchbox",
                    "selector_pattern": None
                },
                "expected_page_change": False
            },
            {
                "step_number": 3,
                "action": "TYPE",
                "description": "Type 'tennis groups' in the search bar",
                "text_input": "tennis groups",
                "target_hints": {
                    "type": "input",
                    "text_contains": ["search"],
                    "placeholder_contains": ["search"],
                    "role": "searchbox",
                    "selector_pattern": None
                },
                "expected_page_change": False
            },
            {
                "step_number": 4,
                "action": "CLICK",
                "description": "Click on 'tennis groups' search suggestion or press Enter to search",
                "text_input": None,
                "target_hints": {
                    "type": "link",
                    "text_contains": ["tennis", "groups"],
                    "placeholder_contains": [],
                    "role": None,
                    "selector_pattern": None
                },
                "expected_page_change": True
            },
            {
                "step_number": 5,
                "action": "CLICK",
                "description": "Click on the 'Groups' filter tab to show only groups in search results",
                "text_input": None,
                "target_hints": {
                    "type": "link",
                    "text_contains": ["groups"],
                    "placeholder_contains": [],
                    "role": None,
                    "selector_pattern": None
                },
                "expected_page_change": True
            },
            {
                "step_number": 6,
                "action": "CLICK",
                "description": "Click on a tennis group from the search results to view it",
                "text_input": None,
                "target_hints": {
                    "type": "link",
                    "text_contains": ["tennis"],
                    "placeholder_contains": [],
                    "role": None,
                    "selector_pattern": None
                },
                "expected_page_change": True
            },
            {
                "step_number": 7,
                "action": "CLICK",
                "description": "Click the 'Join Group' button to request to join the group",
                "text_input": None,
                "target_hints": {
                    "type": "button",
                    "text_contains": ["join", "group"],
                    "placeholder_contains": [],
                    "role": "button",
                    "selector_pattern": None
                },
                "expected_page_change": False
            },
            {
                "step_number": 8,
                "action": "DONE",
                "description": "Successfully requested to join the tennis group! You may need to wait for admin approval.",
                "text_input": None,
                "target_hints": {
                    "type": None,
                    "text_contains": [],
                    "placeholder_contains": [],
                    "role": None,
                    "selector_pattern": None
                },
                "expected_page_change": False
            }
        ],
        "total_steps": 8,
        
        # Quality metrics
        "success_count": 1,
        "failure_count": 0,
        "total_uses": 1,
        "avg_completion_rate": 1.0,
        
        # Metadata
        "user_corrections": [],
        "last_correction_at": None,
        "created_at": now,
        "updated_at": now,
        "last_used_at": now,
        "expires_at": expires_at,
        "original_session_id": "manual-seed"
    }
    
    # Check if already exists
    existing = await collection.find_one({"cache_id": cache_doc["cache_id"]})
    if existing:
        print(f"Cache entry already exists: {cache_doc['cache_id']}")
        # Update instead
        await collection.replace_one({"cache_id": cache_doc["cache_id"]}, cache_doc)
        print("Updated existing entry.")
    else:
        await collection.insert_one(cache_doc)
        print(f"Inserted cache entry: {cache_doc['cache_id']}")
    
    # Verify
    count = await collection.count_documents({})
    print(f"Total cache entries: {count}")
    
    client.close()


if __name__ == "__main__":
    asyncio.run(seed_facebook_tennis_cache())