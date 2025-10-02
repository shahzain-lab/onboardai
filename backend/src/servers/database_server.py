# src/servers/database_server.py

import os
import asyncpg
from typing import Optional, List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL environment variable")

# Define an application-level context to hold shared resources
class AppContext:
    db_pool: asyncpg.pool.Pool

async def create_pool() -> asyncpg.pool.Pool:
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Startup: create pool
    pool = await create_pool()
    appctx = AppContext()
    appctx.db_pool = pool
    try:
        yield appctx
    finally:
        # Shutdown: close pool
        await pool.close()

# Instantiate the server with the lifespan param
mcp = FastMCP(name="TasksDB", lifespan=lifespan)

# Helper to get pool
def get_pool_from_ctx(ctx: Context) -> asyncpg.pool.Pool:
    return ctx.request_context.lifespan_context.db_pool

# Tools — pass ctx: Context to each tool that needs DB
@mcp.tool()
async def get_user(user_id: str, ctx: Context) -> Optional[Dict[str, Any]]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, email, name, role, created_at, updated_at FROM users WHERE user_id = $1",
            user_id,
        )
        return dict(row) if row else None

@mcp.tool()
async def list_users(limit: int = 50, ctx: Context = None) -> List[Dict[str, Any]]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, email, name, role, created_at FROM users ORDER BY created_at DESC LIMIT $1",
            limit,
        )
        return [dict(r) for r in rows]

@mcp.tool()
async def create_user(user_id: str, email: Optional[str] = None,
                      name: Optional[str] = None, role: Optional[str] = None,
                      ctx: Context = None) -> Dict[str, Any]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (user_id, email, name, role, created_at, updated_at)
            VALUES ($1, $2, $3, $4, NOW(), NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET email = EXCLUDED.email,
                  name = EXCLUDED.name,
                  role = EXCLUDED.role,
                  updated_at = NOW()
            RETURNING user_id, email, name, role, created_at, updated_at
            """,
            user_id, email, name, role,
        )
        return dict(row)

@mcp.tool()
async def list_tasks(user_id: Optional[str] = None, status: Optional[str] = None,
                     limit: int = 100, ctx: Context = None) -> List[Dict[str, Any]]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        if user_id and status:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3",
                user_id, status, limit,
            )
        elif user_id:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                user_id, limit,
            )
        elif status:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                status, limit,
            )
        else:
            rows = await conn.fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]

@mcp.tool()
async def get_task(task_id: int, ctx: Context) -> Optional[Dict[str, Any]]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
        return dict(row) if row else None

@mcp.tool()
async def create_task(user_id: str, title: str, description: Optional[str] = None,
                      status: str = "pending", priority: str = "medium",
                      source: Optional[str] = None, source_id: Optional[str] = None,
                      due_date: Optional[str] = None, ctx: Context = None) -> Dict[str, Any]:
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tasks (user_id, title, description, status, priority, source, source_id, due_date, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
            RETURNING *
            """,
            user_id, title, description, status, priority, source, source_id, due_date,
        )
        return dict(row)

@mcp.tool()
async def update_task(task_id: int, title: Optional[str] = None, description: Optional[str] = None,
                      status: Optional[str] = None, priority: Optional[str] = None,
                      due_date: Optional[str] = None, completed_at: Optional[str] = None,
                      ctx: Context = None) -> Optional[Dict[str, Any]]:
    pool = get_pool_from_ctx(ctx)
    updates = []
    args = []
    idx = 1
    for name, val in (("title", title), ("description", description), ("status", status),
                      ("priority", priority), ("due_date", due_date), ("completed_at", completed_at)):
        if val is not None:
            updates.append(f"{name} = ${idx}")
            args.append(val)
            idx += 1
    if not updates:
        return await get_task(task_id, ctx=ctx)

    updates.append("updated_at = NOW()")
    args.append(task_id)
    set_clause = ", ".join(updates)

    async with pool.acquire() as conn:
        await conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ${idx}", *args)
        # reuse get_task
        return await get_task(task_id, ctx=ctx)

@mcp.tool()
async def raw_read(sql: str, limit: int = 100, ctx: Context = None) -> List[Dict[str, Any]]:
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):
        raise ValueError("raw_read only supports SELECT queries")
    pool = get_pool_from_ctx(ctx)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql + f" LIMIT {int(limit)}")
        return [dict(r) for r in rows]

if __name__ == "__main__":
    mcp.run(transport="stdio")
