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
async def kb_store_task(task_id: str, title: str, description: Optional[str],
                        status: str, priority: str, user_id: str,
                        ctx: Context = None) -> Dict[str, Any]:
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
    # integrated embedding: passing `text`
    index.upsert(vectors=[upsert_item])
    return {"id": upsert_item["id"], "metadata": metadata}

@mcp.tool()
async def kb_store_user(user_id: str, name: Optional[str], email: Optional[str],
                        role: Optional[str], status_note: Optional[str],
                        ctx: Context = None) -> Dict[str, Any]:
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
async def kb_store_org(org_id: str, name: str, description: Optional[str],
                       ctx: Context = None) -> Dict[str, Any]:
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
async def kb_query(query: str, top_k: int = 5, ctx: Context = None) -> Dict[str, Any]:
    app = _appctx(ctx)
    index = app.index

    resp = index.query(
        queries=[{"text": query}],
        top_k=top_k,
        include_metadata=True
    )
    # extract matches
    mlist = resp["results"][0].get("matches", []) if "results" in resp else resp.get("matches", [])
    hits = [
        {"id": m.get("id"), "score": m.get("score"), "metadata": m.get("metadata")}
        for m in mlist
    ]
    return {"query": query, "hits": hits}

@mcp.tool()
async def kb_answer_qa(question: str, top_k: int = 5, ctx: Context = None) -> Dict[str, Any]:
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
