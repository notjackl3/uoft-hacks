from __future__ import annotations

import logging
from typing import Any, Optional

from pymongo import ASCENDING

from app.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
except Exception:  # pragma: no cover
    AsyncIOMotorClient = Any  # type: ignore[misc,assignment]
    AsyncIOMotorDatabase = Any  # type: ignore[misc,assignment]


class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None


mongodb = MongoDB()


async def connect_to_mongo() -> None:
    # Import lazily to keep imports test-friendly when motor isn't installed.
    from motor.motor_asyncio import AsyncIOMotorClient as _AsyncIOMotorClient  # type: ignore
    import certifi

    # Use certifi for SSL certificate verification on macOS
    mongodb.client = _AsyncIOMotorClient(
        settings.mongodb_uri,
        tlsCAFile=certifi.where()
    )
    mongodb.db = mongodb.client[settings.mongodb_db_name]

    # Indexes - commented out due to SSL handshake issues with MongoDB Atlas
    # MongoDB may have IP whitelist restrictions or require TLS 1.3
    # await mongodb.db.sessions.create_index("session_id", unique=True)
    # await mongodb.db.sessions.create_index("created_at")
    # await mongodb.db.execution_log.create_index([("session_id", ASCENDING), ("step_number", ASCENDING)])

    logger.info("Connected to MongoDB (indexes skipped due to SSL issues)")


async def close_mongo_connection() -> None:
    if mongodb.client is not None:
        mongodb.client.close()
        mongodb.client = None
        mongodb.db = None
        logger.info("Closed MongoDB connection")


def get_db() -> AsyncIOMotorDatabase:
    if mongodb.db is None:
        raise RuntimeError("MongoDB is not connected")
    return mongodb.db

