"""
Database Connection
===================
Manages async PostgreSQL connection pool for the API.
"""

import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg

from config import get_settings, get_db_logger

logger = get_db_logger()
settings = get_settings()


class DatabasePool:
    """
    Manages PostgreSQL connection pool.
    
    Provides async context manager for connections and
    handles startup/shutdown lifecycle.
    """
    
    _instance: Optional['DatabasePool'] = None
    _pool: Optional[asyncpg.Pool] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the connection pool."""
        return self._pool
    
    @property
    def is_connected(self) -> bool:
        """Check if pool is connected."""
        return self._pool is not None
    
    async def connect(
        self,
        host: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_interval: Optional[int] = None
    ) -> asyncpg.Pool:
        """
        Create connection pool with retry logic.
        
        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_retries: Maximum connection attempts
            retry_interval: Seconds between retries
        
        Returns:
            Connection pool
        """
        if self._pool is not None:
            return self._pool
        
        # Use provided values or defaults from config
        host = host or settings.database.HOST
        database = database or settings.database.NAME
        user = user or settings.database.USER
        password = password or settings.database.PASSWORD
        min_size = min_size or settings.database.MIN_POOL_SIZE
        max_size = max_size or settings.database.MAX_POOL_SIZE
        max_retries = max_retries or settings.database.MAX_RETRIES
        retry_interval = retry_interval or settings.database.RETRY_INTERVAL
        
        for attempt in range(max_retries):
            try:
                self._pool = await asyncpg.create_pool(
                    host=host,
                    database=database,
                    user=user,
                    password=password,
                    min_size=min_size,
                    max_size=max_size
                )
                logger.info("✅ Database connection pool created")
                return self._pool
            
            except (OSError, asyncpg.CannotConnectNowError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database not ready, retrying in {retry_interval}s... "
                        f"({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(
                        f"Failed to connect after {max_retries} attempts: {e}"
                    )
                    raise
            
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                raise
    
    async def disconnect(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("🔒 Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.
        
        Usage:
            async with db_pool.acquire() as conn:
                result = await conn.fetch("SELECT ...")
        """
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args):
        """Execute a query without returning results."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# Global pool instance
db_pool = DatabasePool()


async def get_pool() -> DatabasePool:
    """
    Dependency for FastAPI routes.
    
    Returns:
        Database pool instance
    """
    if not db_pool.is_connected:
        await db_pool.connect()
    return db_pool


async def init_db():
    """Initialize database connection on startup."""
    await db_pool.connect()


async def close_db():
    """Close database connection on shutdown."""
    await db_pool.disconnect()
