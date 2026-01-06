"""
Redis-backed Rate Limiter and Cache for Artemis Signals
- Limits API calls per user/IP/key
- Provides distributed cache for backend
- For Shivaansh & Krishaansh — this line pays your fees!
"""
import redis
import time
import os
from utils.logger import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisRateLimiter:
    def __init__(self, redis_url=REDIS_URL, prefix="rate_limit:"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

    def is_allowed(self, key: str, max_calls: int, period: int) -> bool:
        """
        Returns True if the action is allowed (not rate limited), else False.
        Args:
            key: Unique identifier (user_id, IP, endpoint)
            max_calls: Max allowed calls in period
            period: Period in seconds
        """
        redis_key = f"{self.prefix}{key}"
        now = int(time.time())
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(redis_key, 0, now - period)
        pipeline.zadd(redis_key, {str(now): now})
        pipeline.zcard(redis_key)
        pipeline.expire(redis_key, period)
        _, _, count, _ = pipeline.execute()
        allowed = count <= max_calls
        if not allowed:
            logger.warning(f"[RATE LIMIT] Blocked {key}: {count}/{max_calls} in {period}s")
        return allowed


    def get_count(self, key: str, period: int) -> int:
        redis_key = f"{self.prefix}{key}"
        now = int(time.time())
        self.redis.zremrangebyscore(redis_key, 0, now - period)
        return self.redis.zcard(redis_key)

    def get_all_usage(self, period: int = 3600) -> dict:
        """
        Returns usage counts for all keys in the current period.
        """
        usage = {}
        try:
            for key in self.redis.scan_iter(f"{self.prefix}*"):
                count = self.get_count(key.replace(self.prefix, ""), period)
                usage[key.replace(self.prefix, "")] = count
        except Exception as e:
            logger.error(f"[USAGE_METERING] Failed to fetch usage: {e}")
        return usage

    def reset(self, key: str):
        redis_key = f"{self.prefix}{key}"
        self.redis.delete(redis_key)

# Simple Redis cache for backend (key-value, with TTL)
class RedisCache:
    def __init__(self, redis_url=REDIS_URL, prefix="cache:"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

    def set(self, key: str, value, ttl: int = 300):
        redis_key = f"{self.prefix}{key}"
        self.redis.set(redis_key, value, ex=ttl)

    def get(self, key: str):
        redis_key = f"{self.prefix}{key}"
        return self.redis.get(redis_key)

    def delete(self, key: str):
        redis_key = f"{self.prefix}{key}"
        self.redis.delete(redis_key)
