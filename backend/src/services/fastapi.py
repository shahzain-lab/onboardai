# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

import json
import os
import hmac
import hashlib
import httpx
import time
import asyncio
import logging
import urllib.parse
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
from config.slack_client import slack_client
from fastapi.middleware.cors import CORSMiddleware
from services.workflow_graph import workflow_graph
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config.pydantic_models import SlackEventRequest, SlackCommandRequest, StandupRequest, MeetingRequest, QARequest, OnboardingRequest, TaskUpdate
from config.env_config import config as env

SLACK_SIGNING_SECRET = env.SLACK_SIGNING_SECRET

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting AI Workplace Assistant...")
    print("✅ AutoGen agents initialized")
    print("✅ LangGraph workflow ready")
    yield
    # Shutdown
    print("👋 Shutting down AI Workplace Assistant...")

app = FastAPI(
    title="AI Workplace Assistant",
    description="Slack/Discord bot with AutoGen multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTH DEPENDENCIES
# ============================================================================
security = HTTPBearer()
log = logging.getLogger("slack-webhook")

async def post_to_slack_response_url(response_url: str, payload: Dict[str, Any]):
    """
    Post JSON payload to Slack response_url and log result.
    Payload should include "text" and optionally "blocks".
    """
    # Ensure response_type in payload so message is visible to channel
    payload_to_send = dict(payload)  # shallow copy
    # Prefer final to be in_channel unless caller wants ephemeral
    if "response_type" not in payload_to_send:
        payload_to_send["response_type"] = "in_channel"

    log.info("Posting result to Slack response_url...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(response_url, json=payload_to_send)
            log.info("Slack POST status: %s, body: %s", r.status_code, r.text)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.exception("Failed to post to Slack response_url: %s", e)
            # optionally: fallback to Slack Web API using BOT token if you have it
            # or save result to DB for retry
            return False
    return True


def format_slack_response(
    command: str,
    user_name: str,
    user_id: str,
    text: str,
    workflow_result: dict | None = None
) -> dict:
    workflow_result = workflow_result or {}
    metadata = workflow_result.get("result", {}) or {}

    # 1️⃣ Prefer explicit final_summary
    final_summary = workflow_result.get("final_summary")

    # 2️⃣ Fallback: nested summaries
    standup_summary = None
    if isinstance(metadata.get("standup_result"), dict):
        standup_summary = metadata["standup_result"].get("summary")

    # 3️⃣ Fallback: plain result
    def get_result(key: str, fallback: str = "") -> str:
        if isinstance(metadata.get(key), dict):
            return metadata[key].get("result", fallback).strip()
        return fallback

    # Pick the best available summary
    summary_text = final_summary or standup_summary or get_result("standup_result", "No summary available.")

    # --- Format based on command ---
    if command == "/standup":
        body = f"📌 *Standup Summary* for *{user_name}*\n{summary_text}"
    elif command == "/onboard":
        body = f"🚀 Onboarding update for *{user_name}*\n{final_summary or get_result('onboarding_result', 'Onboarding started.')}"
    elif command == "/ask":
        body = f"💡 Question from *{user_name}*:\n> {text}\n\n{final_summary or get_result('qa_result', 'Answer will be available soon.')}"
    elif command == "/meeting":
        body = f"📅 Meeting Summary:\n{final_summary or get_result('meeting_result', 'Meeting summary not found.')}"
    elif command == "/transcribe":
        body = f"📝 Transcript:\n{final_summary or get_result('transcription_result', 'Transcript unavailable.')}"
    else:
        body = f"⚠️ Unknown command: `{command}` by *{user_name}* ({user_id})"

    # --- Footer (context info) ---
    conv_id = workflow_result.get("conversation_id") or metadata.get("conversation_id")
    agent_used = metadata.get("agent_used") or workflow_result.get("agent_used")

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": body}}]

    footer_parts = []
    if agent_used:
        footer_parts.append(f"Processed by: `{agent_used}`")
    if conv_id:
        footer_parts.append(f"Conversation: `{conv_id}`")
    if footer_parts:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " • ".join(footer_parts)}]
        })

    return {"text": body, "blocks": blocks}

# ---------- workflow runner ----------

# IMPORTANT: import your actual workflow_graph object here
# from services.workflow import workflow_graph
# For this example, assume workflow_graph exists in the global scope.
# If it is in a module, import it at top of file.

async def run_workflow_and_post_result(command: str, text: str, user_name: str, user_id: str, response_url: str):
    """
    Runs the workflow (may take time), formats result, and posts to Slack response_url.
    This function is safe to cancel on shutdown.
    """
    try:
        log.info("Running workflow for %s / %s", command, user_id)

        # map slash command to workflow type (remove leading '/')
        workflow_type = command.lstrip("/")

        # call your langgraph workflow (adjust call if different)
        try:
            workflow_result = await workflow_graph.execute_workflow(
                workflow_type,
                {"command_text": text, "user_id": user_id},
                user_id
            )
        except Exception as e:
            log.exception("Workflow execution failed: %s", e)
            workflow_result = {"metadata": {"error": str(e)}}

        log.info("Workflow result: %s", workflow_result)

        # Build slack payload using the formatter
        slack_payload = format_slack_response(command, user_name, user_id, text, workflow_result)
        log.info("Posting formatted payload to slack: %s", slack_payload)

        # Post to response_url and log status
        success = await post_to_slack_response_url(response_url, slack_payload)
        if not success:
            log.error("Posting to Slack failed; payload saved for retry (implement retry logic).")

    except asyncio.CancelledError:
        log.warning("Background workflow task cancelled (shutdown).")
        # do not re-raise
        return
    except Exception:
        log.exception("Unhandled error in run_workflow_and_post_result")
        # optionally post an error to Slack using response_url
        try:
            await post_to_slack_response_url(response_url, {
                "response_type": "ephemeral",
                "text": "⚠️ Failed to process your request. Try again later."
            })
        except Exception:
            pass

class SlackChallenge(BaseModel):
    token: str
    challenge: str
    type: str

# Helper to cache body so multiple reads don't consume
async def get_cached_body(request: Request) -> bytes:
    if not hasattr(request, "_cached_body"):
        request._cached_body = await request.body()
    return request._cached_body

def verify_slack_request(request: Request, body: bytes):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")

    if not timestamp or not slack_signature:
        raise HTTPException(status_code=400, detail="Missing Slack headers")

    # Prevent replay attacks (within 5 minutes)
    try:
        req_ts = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    if abs(time.time() - req_ts) > 60 * 5:
        raise HTTPException(status_code=400, detail="Request too old")

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    my_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode("utf-8"),
        sig_basestring,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(my_signature, slack_signature):
        raise HTTPException(status_code=400, detail="Invalid Slack signature")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple JWT validation (implement your own logic)"""
    token = credentials.credentials
    # Implement your JWT validation logic here
    return {"user_id": "user123", "email": "user@example.com"}

# ============================================================================
# WEBHOOK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Workplace Assistant is running!",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "slack_commands": "/webhook/slack/commands",
            "slack_events": "/webhook/slack/events", 
            "health": "/api/health",
            "qa": "/api/qa/query"
        }
    }

@app.post("/webhook/slack/events")
async def slack_events(event_data: SlackEventRequest, background_tasks: BackgroundTasks):
    """Handle Slack event subscriptions"""
    try:
        event = event_data.event
        
        if event.get("type") == "message" and not event.get("bot_id"):
            # Process user message
            background_tasks.add_task(
                process_slack_message,
                event.get("channel"),
                event.get("user"),
                event.get("text", "")
            )
        
        return {"challenge": event_data.dict().get("challenge", "ok")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/slack/events")
async def slack_events(challenge_data: SlackChallenge):
    return {"challenge": challenge_data.challenge}


@app.post("/webhook/slack/commands")
async def slack_commands(request: Request, background_tasks: BackgroundTasks):
    # 1) get body once and verify
    body = await get_cached_body(request)
    verify_slack_request(request, body)

    # 2) parse form data
    form = urllib.parse.parse_qs(body.decode("utf-8"))
    command = form.get("command", [""])[0]
    text = form.get("text", [""])[0]
    user_id = form.get("user_id", [""])[0]
    user_name = form.get("user_name", [""])[0]
    response_url = form.get("response_url", [""])[0]

    log.info("Slash command received: %s by %s", command, user_name)

    # 3) immediate ack (must return within 3s)
    ack = {"response_type": "ephemeral", "text": f"⏳ Working on {command}... I'll post the result here shortly."}

    # 4) schedule background job
    background_tasks.add_task(run_workflow_and_post_result, command, text, user_name, user_id, response_url)

    return ack


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def process_slack_message(channel: str, user: str, text: str):
    """Process Slack message in background"""
    try:
        # Determine if it's a question or command
        if "?" in text or text.lower().startswith(("what", "how", "when", "where", "why")):
            result = await workflow_graph.execute_workflow(
                "qa",
                {"question": text, "user_id": user, "channel": channel}
            )
        else:
            result = await workflow_graph.execute_workflow(
                "general",
                {"message": text, "user_id": user, "channel": channel}
            )
        
        # Send response back to Slack
        response_text = json.dumps(result.get("metadata", {}), indent=2)
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"🤖 AI Assistant Response:\n```{response_text}```"
        )
    except Exception as e:
        print(f"Error processing Slack message: {e}")
