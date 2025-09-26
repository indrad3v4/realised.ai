"""
Database adapter for Team-Realized platform
Simple SQLite + Redis connection for hackathon MVP
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import sqlite3
import aiosqlite
import redis.asyncio as redis
from pathlib import Path

from ..core.entities import (
    AffordableProperty, Opportunity, Transaction, 
    Portfolio, MarketAnalysis, PropertyType, TransactionStatus
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for SQLite + Redis"""

    def __init__(self):
        self.db_path = "team_realized.db"
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return

        try:
            await self._init_sqlite()
            await self._init_redis()
            self._initialized = True
            logger.info("Database service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _init_sqlite(self):
        """Initialize SQLite database and create tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Properties table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS properties (
                    id TEXT PRIMARY KEY,
                    city TEXT NOT NULL,
                    address TEXT NOT NULL,
                    property_type TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    size_sqm REAL NOT NULL,
                    rooms INTEGER NOT NULL,
                    year_built INTEGER NOT NULL,
                    price_per_sqm REAL NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    neighborhood TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Transactions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    opportunity_id TEXT NOT NULL,
                    property_id TEXT NOT NULL,
                    investment_amount REAL NOT NULL,
                    ownership_percentage REAL NOT NULL,
                    status TEXT NOT NULL,
                    payment_method TEXT NOT NULL,
                    payment_reference TEXT NOT NULL,
                    blockchain_network TEXT DEFAULT 'SOLANA',
                    token_mint_address TEXT,
                    certificate_token_id TEXT,
                    fees_paid REAL DEFAULT 0.0,
                    initiated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    failed_at TIMESTAMP
                )
            """)

            await db.commit()
            logger.info("SQLite database initialized")

    async def _init_redis(self):
        """Initialize Redis connection with fallback"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without Redis cache.")
            self.redis_client = None

    async def save_transaction(self, transaction: Transaction) -> bool:
        """Save transaction to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO transactions (
                        id, user_id, opportunity_id, property_id, investment_amount,
                        ownership_percentage, status, payment_method, payment_reference,
                        blockchain_network, token_mint_address, certificate_token_id,
                        fees_paid, initiated_at, completed_at, failed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction.transaction_id, transaction.user_id,
                    transaction.opportunity_id, transaction.property_id,
                    transaction.investment_amount, transaction.ownership_percentage,
                    transaction.status.value, transaction.payment_method,
                    transaction.payment_reference, transaction.blockchain_network,
                    transaction.token_mint_address, transaction.certificate_token_id,
                    transaction.fees_paid, transaction.initiated_at.isoformat(),
                    transaction.completed_at.isoformat() if transaction.completed_at else None,
                    transaction.failed_at.isoformat() if transaction.failed_at else None
                ))
                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save transaction {transaction.transaction_id}: {e}")
            return False


# Global database instance
database_service = DatabaseService()
