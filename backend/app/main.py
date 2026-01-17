from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import close_mongo_connection, connect_to_mongo
from app.routes import session
from app.utils.rate_limiter import get_rate_limit_status


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

        @app.on_event("shutdown")
        async def shutdown_event():
            await close_mongo_connection()

    app.include_router(session.router, prefix="/api/session", tags=["session"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/debug/rate-limit")
    async def rate_limit_status():
        """Check current rate limit status for Gemini API calls."""
        return get_rate_limit_status()

    return app


app = create_app(with_db=True)
