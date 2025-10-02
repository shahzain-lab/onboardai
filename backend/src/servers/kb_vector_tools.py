# src/servers/kb_vector_tools.py
import os
import time
from typing import AsyncIterator, Optional, List, Dict, Any
from contextlib import asynccontextmanager

from pinecone import Pinecone  
from mcp.server.fastmcp import FastMCP, Context

# --------------- Config ---------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "onboardai-kb")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is required")

# --------------- App context ---------------
class AppContext:
    index: Any  # the index client

# --------------- Lifespan ---------------
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Create Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX):
        pc.create_index_for_model(
            name=PINECONE_INDEX,
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "text"}
            }
        )

    # Get index client
    idx = pc.Index(PINECONE_INDEX)
    ctx = AppContext()
    ctx.index = idx
    try:
        yield ctx
    finally:
        # no explicit teardown needed
        pass

mcp = FastMCP(name="VectorKBIntegrated", lifespan=lifespan)

# --------------- Helpers ---------------
def _appctx(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context

def _vector_id(kind: str, source_id: str) -> str:
    return f"{kind}:{source_id}"

# --------------- Tools ---------------

@mcp.tool()
async def kb_store_task(
    task_id: str,
    title: str,
    description: Optional[str],
    status: str,
    priority: str,
    user_id: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Store a task in the vector knowledge base using integrated embedding.

    Args:
        task_id: A unique identifier for the task (string).
        title: The title or summary of the task.
        description: Longer description or details (optional).
        status: Current status of the task (e.g. "pending", "completed").
        priority: Priority level (e.g. "low", "medium", "high", "urgent").
        user_id: The ID of the user assigned to the task.
    Returns:
        A dict containing the vector ID and metadata stored.
    """
    app = _appctx(ctx)
    index = app.index

    text = f"Task: {title}\n{description or ''}\nStatus: {status}, Priority: {priority}, Assigned: {user_id}"
    metadata = {
        "kind": "task",
        "task_id": task_id,
        "title": title,
        "status": status,
        "priority": priority,
        "user_id": user_id,
        "stored_at": int(time.time())
    }
    upsert_item = {
        "id": _vector_id("task", task_id),
        "metadata": metadata,
        "text": text
    }
    index.upsert(vectors=[upsert_item])
    return {"id": upsert_item["id"], "metadata": metadata}

@mcp.tool()
async def kb_store_user(
    user_id: str,
    name: Optional[str],
    email: Optional[str],
    role: Optional[str],
    status_note: Optional[str],
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Store a user profile and status information in vector KB.

    Args:
        user_id: Unique identifier for the user (string).
        name: Full name of the user (optional).
        email: Email address (optional).
        role: The user's role or title (optional).
        status_note: A short note describing what the user is working on.
    Returns:
        A dict containing the vector ID and metadata stored.
    """
    app = _appctx(ctx)
    index = app.index

    text = f"User: {name or user_id}, Role: {role}, Email: {email}, Status: {status_note}"
    metadata = {
        "kind": "user",
        "user_id": user_id,
        "name": name,
        "email": email,
        "role": role,
        "status_note": status_note,
        "stored_at": int(time.time())
    }
    upsert_item = {
        "id": _vector_id("user", user_id),
        "metadata": metadata,
        "text": text
    }
    index.upsert(vectors=[upsert_item])
    return {"id": upsert_item["id"], "metadata": metadata}

@mcp.tool()
async def kb_store_org(
    org_id: str,
    name: str,
    description: Optional[str],
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Store organization description and metadata into the KB.

    Args:
        org_id: Identifier for the organization (string).
        name: Human-readable name of the organization.
        description: Longer description or context about the org.
    Returns:
        A dict containing the vector ID and metadata stored.
    """
    app = _appctx(ctx)
    index = app.index

    text = f"Organization: {name}\n{description or ''}"
    metadata = {
        "kind": "org",
        "org_id": org_id,
        "name": name,
        "description": description,
        "stored_at": int(time.time())
    }
    upsert_item = {
        "id": _vector_id("org", org_id),
        "metadata": metadata,
        "text": text
    }
    index.upsert(vectors=[upsert_item])
    return {"id": upsert_item["id"], "metadata": metadata}

@mcp.tool()
async def kb_query(
    query: str,
    top_k: int = 5,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Perform a semantic search in the KB.

    Args:
        query: Query text to search.
        top_k: Number of top results to return (default 5).
    Returns:
        A dict with the original query and an array of hits, each with id, score, and metadata.
    """
    app = _appctx(ctx)
    index = app.index

    resp = index.query(
        queries=[{"text": query}],
        top_k=top_k,
        include_metadata=True
    )
    mlist = resp["results"][0].get("matches", []) if "results" in resp else resp.get("matches", [])
    hits = [
        {"id": m.get("id"), "score": m.get("score"), "metadata": m.get("metadata")}
        for m in mlist
    ]
    return {"query": query, "hits": hits}

@mcp.tool()
async def kb_answer_qa(
    question: str,
    top_k: int = 5,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Retrieve relevant entries for a question and provide context for answering.

    Args:
        question: The question or prompt to answer.
        top_k: Number of related context items to fetch (default 5).
    Returns:
        A dict containing:
          - question: the input question
          - context: a textual summary of the relevant hits
          - hits: array of the matched metadata
    """
    qres = await kb_query.__wrapped__(question, top_k=top_k, ctx=ctx)
    hits = qres["hits"]
    snippets = []
    for h in hits:
        md = h.get("metadata", {})
        kind = md.get("kind")
        if kind == "task":
            snippets.append(f"- Task {md.get('title')} (status: {md.get('status')})")
        elif kind == "user":
            snippets.append(f"- User {md.get('name')} (role: {md.get('role')}) {md.get('status_note')}")
        elif kind == "org":
            snippets.append(f"- Org {md.get('name')}: {md.get('description')}")
    context_text = "\n".join(snippets)
    return {
        "question": question,
        "context": context_text,
        "hits": hits
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
