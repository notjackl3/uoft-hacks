from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import close_mongo_connection, connect_to_mongo, get_db
from app.routes import session
from app.routes.companies import router as companies_router
from app.routes.commerce import router as commerce_router
from app.routes import cache as cache_routes
from app.utils.rate_limiter import get_rate_limit_status
from app.services.backboard_ai import backboard_ai
from app.services.graph import graph_service
from app.services.cache_service import ensure_cache_indexes


def create_app(with_db: bool = True) -> FastAPI:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(title="Universal On-Screen Tutor API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if with_db:

        @app.on_event("startup")
        async def startup_event():
            await connect_to_mongo()
            # Initialize Neo4j schema
            try:
                if graph_service.verify_connectivity():
                    graph_service.setup_schema()
                    graph_service.setup_vector_index()
            except Exception as e:
                logging.warning(f"Neo4j setup skipped: {e}")
            # Initialize cache indexes
            try:
                db = get_db()
                await ensure_cache_indexes(db)
            except Exception as e:
                logging.warning(f"Failed to initialize cache indexes (non-fatal): {e}")

        @app.on_event("shutdown")
        async def shutdown_event():
            await close_mongo_connection()
            graph_service.close()

    app.include_router(session.router, prefix="/api/session", tags=["session"])
    app.include_router(companies_router, prefix="/api/companies", tags=["companies"])
    app.include_router(commerce_router, prefix="/api/commerce", tags=["commerce"])
    app.include_router(cache_routes.router, prefix="/api", tags=["cache"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/debug/rate-limit")
    async def rate_limit_status():
        """Check current rate limit status for Gemini API calls."""
        return get_rate_limit_status()

    @app.get("/api/backboard/stats")
    async def backboard_stats():
        """Get Backboard.io model usage statistics (demonstrates multi-model switching)"""
        return backboard_ai.get_model_stats()

    @app.post("/api/backboard/learn")
    async def learn_pattern(user_id: str, interaction_data: dict):
        """Learn from user interaction (adaptive memory feature)"""
        result = await backboard_ai.learn_pattern(user_id, interaction_data)
        return result

    return app


app = create_app(with_db=True)
