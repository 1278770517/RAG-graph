#
# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# import psycopg
#
#
# async def diagnose():
#     dbs = [
#         "postgresql://postgres:123456@localhost:5432/ragshort_checkpoint",
#         "postgresql://postgres:123456@localhost:5432/raglong_memory",
#     ]
#     for uri in dbs:
#         try:
#             conn = await psycopg.AsyncConnection.connect(uri)
#             cur = await conn.execute("SELECT current_database()")
#             dbname = (await cur.fetchone())[0]
#             print(f"✅ {dbname} - 连接正常")
#             await conn.close()
#         except Exception as e:
#             print(f"❌ {uri.split('/')[-1]} - 错误: {e}")
#
# asyncio.run(diagnose())
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def test():
    pool = AsyncConnectionPool(
        conninfo="postgresql://postgres:123456@localhost:5432/ragshort_checkpoint",
        max_size=20,
    )
    saver = AsyncPostgresSaver(pool)
    try:
        await saver.setup()
        print("✅ AsyncPostgresSaver 支持 AsyncConnectionPool")
    except Exception as e:
        print(f"❌ 不支持 Pool: {type(e).__name__}: {e}")
    finally:
        pool.close()

asyncio.run(test())