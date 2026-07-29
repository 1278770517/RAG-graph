import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import Dict




class AsyncSessionManager:
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._checkpoint_pool = None
        self._memory_pool = None
        self._checkpoint = None
        self._memory = None

    async def connect(self):
        """程序启动时调用一次"""
        # checkpoint 连接池
        self._checkpoint_pool = AsyncConnectionPool(
            conninfo="postgresql://postgres:123456@localhost:5432/ragshort_checkpoint",
            max_size=20,
            open=False,  # 禁止构造函数自动打开（已废弃）
        )
        await self._checkpoint_pool.open()   # ✅ 必须显式 await open()
        self._checkpoint = AsyncPostgresSaver(self._checkpoint_pool)
        await self._checkpoint.setup()       # 建表（首次运行需要）

        # memory 连接池
        self._memory_pool = AsyncConnectionPool(
            conninfo="postgresql://postgres:123456@localhost:5432/raglong_memory",
            max_size=20,
            open=False,
        )
        await self._memory_pool.open()       # ✅ 必须显式 await open()
        self._memory = AsyncPostgresStore(self._memory_pool)
        await self._memory.setup()           # 建表（首次运行需要）

        print(f"Checkpoint: {self._checkpoint}")
        print(f"Memory: {self._memory}")

    async def get_memory(self):
        return self._checkpoint, self._memory

    async def disconnect(self):
        """程序退出时调用"""
        if self._checkpoint_pool:
            await self._checkpoint_pool.close()   # ✅ 必须 await
        if self._memory_pool:
            await self._memory_pool.close()       # ✅ 必须 await
from typing import Dict, Tuple, Optional
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool  # 注意：同步版本


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._checkpoint_pool: Optional[ConnectionPool] = None
        self._memory_pool: Optional[ConnectionPool] = None
        self._checkpoint: Optional[PostgresSaver] = None
        self._memory: Optional[PostgresStore] = None

    def connect(self) -> None:
        """同步连接"""
        self._checkpoint_pool = ConnectionPool(
            conninfo="postgresql://postgres:123456@localhost:5432/ragshort_checkpoint",
            max_size=20,
            open=False,
        )
        self._checkpoint_pool.open()          # 同步 open，无需 await
        self._checkpoint = PostgresSaver(self._checkpoint_pool)
        self._checkpoint.setup()              # 同步 setup

        self._memory_pool = ConnectionPool(
            conninfo="postgresql://postgres:123456@localhost:5432/raglong_memory",
            max_size=20,
            open=False,
        )
        self._memory_pool.open()
        self._memory = PostgresStore(self._memory_pool)
        self._memory.setup()

        print(f"Checkpoint: {self._checkpoint}")
        print(f"Memory: {self._memory}")

    def get_memory(self) -> Tuple[Optional[PostgresSaver], Optional[PostgresStore]]:
        return self._checkpoint, self._memory

    def disconnect(self) -> None:
        """同步断开"""
        if self._checkpoint_pool:
            self._checkpoint_pool.close()     # 同步 close
        if self._memory_pool:
            self._memory_pool.close()