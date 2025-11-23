from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from fastapi import FastAPI
from pymongo import ASCENDING, DESCENDING


# Load environment variables from a .env file if present
load_dotenv()

MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "crisis_connect")


async def init_mongo(app: FastAPI) -> None:
    """Initialize a single Mongo client and attach it to the app state.

    This should be called once on FastAPI startup.
    """
    client = AsyncIOMotorClient(MONGODB_URI)

    # Verify connection early to fail fast
    await client.admin.command("ping")

    app.state.mongo_client = client
    app.state.db = client[MONGODB_DB]


async def close_mongo(app: FastAPI) -> None:
    """Close the Mongo client on shutdown."""
    client: AsyncIOMotorClient | None = getattr(app.state, "mongo_client", None)
    if client is not None:
        client.close()
        app.state.mongo_client = None
        app.state.db = None


def get_db(app: FastAPI) -> AsyncIOMotorDatabase:
    """Convenience accessor for the database bound to the app."""
    db: AsyncIOMotorDatabase | None = getattr(app.state, "db", None)
    if db is None:
        raise RuntimeError("MongoDB is not initialized. Call init_mongo() on startup.")
    return db


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes to optimize common queries and avoid duplicates.

    This is idempotent; creating an existing index is a no-op.
    """
    await db["weather_data"].create_index([("location", ASCENDING), ("timestamp", DESCENDING)])
    # Prevent duplicate weather entries per location+timestamp
    try:
        await db["weather_data"].create_index(
            [("location", ASCENDING), ("timestamp", ASCENDING)], name="uniq_location_timestamp", unique=True
        )
    except Exception:
        # Unique creation can fail if duplicates exist already; keep non-unique index above
        pass

    await db["predictions"].create_index([("location", ASCENDING), ("timestamp", DESCENDING)])
    try:
        await db["predictions"].create_index(
            [("location", ASCENDING), ("timestamp", ASCENDING)], name="uniq_pred_location_timestamp", unique=True
        )
    except Exception:
        pass

    await db["alerts"].create_index([("timestamp", DESCENDING)])
    await db["alerts"].create_index([("location", ASCENDING)])
    try:
        await db["alerts"].create_index(
            [("location", ASCENDING), ("timestamp", ASCENDING)], name="uniq_alert_location_timestamp", unique=False
        )
    except Exception:
        pass

    await db["historical_events"].create_index([("location", ASCENDING)])
    await db["historical_summary"].create_index([("location", ASCENDING)], name="idx_hist_summary_location")
    await db["location_risks"].create_index([("created_at", DESCENDING)])
