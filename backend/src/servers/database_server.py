# mcp_server.py
import os
from typing import Optional, List, Dict, Any
import asyncpg
from mcp.server.fastmcp import FastMCP

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL environment variable (Neon/Postgres connection string)")

mcp = FastMCP(name="TasksDB")

# ---------- Database helpers ----------
async def create_pool():
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)

_pool: Optional[asyncpg.pool.Pool] = None

async def get_pool() -> asyncpg.pool.Pool:
    global _pool
    if _pool is None:
        _pool = await create_pool()
    return _pool

# ---------- Tools: Users ----------
@mcp.tool()
async def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by user_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, email, name, role, created_at, updated_at FROM users WHERE user_id = $1",
            user_id,
        )
        return dict(row) if row else None

@mcp.tool()
async def list_users(limit: int = 50) -> List[Dict[str, Any]]:
    """List users (most recent first)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, email, name, role, created_at FROM users ORDER BY created_at DESC LIMIT $1",
            limit,
        )
        return [dict(r) for r in rows]

@mcp.tool()
async def create_user(
    user_id: str, email: Optional[str] = None, name: Optional[str] = None, role: Optional[str] = None
) -> Dict[str, Any]:
    """Create or upsert a user. Returns the user row."""
    pool = await get_pool()
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
            user_id,
            email,
            name,
            role,
        )
        return dict(row)

# ---------- Tools: Tasks ----------
@mcp.tool()
async def list_tasks(user_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List tasks, optionally filtered by user_id and/or status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if user_id and status:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3",
                user_id,
                status,
                limit,
            )
        elif user_id:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                user_id,
                limit,
            )
        elif status:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                status,
                limit,
            )
        else:
            rows = await conn.fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]

@mcp.tool()
async def get_task(task_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific task by its numeric id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
        return dict(row) if row else None

@mcp.tool()
async def create_task(
    user_id: str,
    title: str,
    description: Optional[str] = None,
    status: str = "pending",
    priority: str = "medium",
    source: Optional[str] = None,
    source_id: Optional[str] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new task. due_date should be ISO timestamp if provided."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tasks (user_id, title, description, status, priority, source, source_id, due_date, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
            RETURNING *
            """,
            user_id,
            title,
            description,
            status,
            priority,
            source,
            source_id,
            due_date,
        )
        return dict(row)

@mcp.tool()
async def update_task(
    task_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update allowed fields on a task."""
    updates = []
    args = []
    idx = 1
    for name, val in (
        ("title", title),
        ("description", description),
        ("status", status),
        ("priority", priority),
        ("due_date", due_date),
        ("completed_at", completed_at),
    ):
        if val is not None:
            updates.append(f"{name} = ${idx}")
            args.append(val)
            idx += 1
    if not updates:
        return await get_task(task_id)

    updates.append("updated_at = NOW()")
    args.append(task_id)
    set_clause = ", ".join(updates)

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ${idx}", *args)
        return await get_task(task_id)

# ---------- Tool: raw_read (for advanced debug only) ----------
@mcp.tool()
async def raw_read(sql: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Execute a read-only SELECT query. Only supports SELECT for safety."""
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):
        raise ValueError("raw_read only supports SELECT queries")
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql + f" LIMIT {int(limit)}")
        return [dict(r) for r in rows]

# ---------- Server lifecycle ----------
@mcp.on_startup()
async def _startup() -> None:
    await get_pool()  # just ensure pool ready

@mcp.on_shutdown()
async def _shutdown() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

# ---------- Run ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")
