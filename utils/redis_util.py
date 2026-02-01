import redis
from typing import Any, Dict, List, Optional


class RedisUtil:
    _pool = None
    _host = 'localhost'
    _port = 6379
    _db = 0

    @classmethod
    def init(cls, host: str = 'localhost', port: int = 6379, db: int = 0):
        cls._host = host
        cls._port = port
        cls._db = db
        cls._pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)

    @classmethod
    def _get_connection(cls) -> redis.Redis:
        if cls._pool is None:
            cls.init()
        return redis.Redis(connection_pool=cls._pool)

    @classmethod
    def ping(cls) -> bool:
        return cls._get_connection().ping()

    @classmethod
    def exists(cls, key: str) -> bool:
        return cls._get_connection().exists(key) > 0

    @classmethod
    def delete(cls, *keys: str) -> int:
        return cls._get_connection().delete(*keys)

    @classmethod
    def set(cls, key: str, value: Any, ex: Optional[int] = None) -> bool:
        return cls._get_connection().set(key, value, ex=ex)

    @classmethod
    def hget(cls, name: str, key: str) -> Optional[str]:
        return cls._get_connection().hget(name, key)

    @classmethod
    def hset(cls, name: str, key: str, value: Any) -> int:
        return cls._get_connection().hset(name, key, value)

    @classmethod
    def hgetall(cls, name: str) -> Dict[str, str]:
        return cls._get_connection().hgetall(name)

    @classmethod
    def hdel(cls, name: str, *keys: str) -> int:
        if not keys:
            return 0
        return cls._get_connection().hdel(name, *keys)

    @classmethod
    def zadd(cls, name: str, mapping: Dict[str, float]) -> int:
        return cls._get_connection().zadd(name, mapping)

    @classmethod
    def zrangebyscore(cls, name: str, min: str, max: str) -> List[str]:
        return cls._get_connection().zrangebyscore(name, min, max)

    @classmethod
    def sadd(cls, name: str, *values: str) -> int:
        return cls._get_connection().sadd(name, *values)

    @classmethod
    def smembers(cls, name: str):
        return cls._get_connection().smembers(name)

    @classmethod
    def flushdb(cls) -> bool:
        return cls._get_connection().flushdb()
