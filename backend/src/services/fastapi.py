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


@app.post("/webhook/discord/events")
async def discord_events(event_data: Dict[str, Any]):
    """Handle Discord gateway events"""
    try:
        # Process Discord messages, commands, etc.
        return {"status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/google/calendar")
async def google_calendar_webhook(event_data: Dict[str, Any]):
    """Handle Google Calendar events"""
    try:
        # Process calendar events (meeting scheduled with Meet link)
        return {"status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/google/meet")
async def google_meet_webhook(event_data: Dict[str, Any]):
    """Handle Google Meet lifecycle events"""
    try:
        # Process meeting start/end, participant events
        return {"status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/google/recording")
async def google_recording_webhook(event_data: Dict[str, Any]):
    """Handle Google Meet recording availability"""
    try:
        recording_url = event_data.get("recording_url")
        meeting_id = event_data.get("meeting_id")
        
        # Trigger transcription workflow
        result = await workflow_graph.execute_workflow(
            "transcription",
            {"recording_url": recording_url, "meeting_id": meeting_id}
        )
        
        return {"status": "processing", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS
# ============================================================================

# Auth & User Management
@app.post("/api/auth/login")
async def login(credentials: Dict[str, str]):
    """Authenticate users (OAuth/JWT)"""
    # Implement your auth logic here
    return {"access_token": "fake_token", "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(user=Depends(get_current_user)):
    """Logout user"""
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
async def get_current_user_profile(user=Depends(get_current_user)):
    """Get current user profile"""
    return user

# Standups
@app.post("/api/standup/start")
async def start_standup(standup_data: StandupRequest, user=Depends(get_current_user)):
    """Trigger standup collection manually"""
    try:
        result = await workflow_graph.execute_workflow(
            "standup",
            standup_data.dict(),
            standup_data.user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/standup/report/{date}")
async def get_standup_report(date: str, user=Depends(get_current_user)):
    """Fetch daily standup summary"""
    try:
        # Implement database query logic
        return {"date": date, "summary": "Standup report for " + date}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/standup/history")
async def get_standup_history(user=Depends(get_current_user)):
    """List past standup reports"""
    try:
        # Implement database query logic
        return {"reports": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Meetings
@app.post("/api/meeting/process")
async def process_meeting(meeting_data: MeetingRequest, user=Depends(get_current_user)):
    """Manually trigger transcription + summarization"""
    try:
        result = await workflow_graph.execute_workflow(
            "meeting",
            meeting_data.dict()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/meeting/summary/{meeting_id}")
async def get_meeting_summary(meeting_id: str, user=Depends(get_current_user)):
    """Fetch meeting summary & action items"""
    try:
        # Implement database query logic
        return {"meeting_id": meeting_id, "summary": "Meeting summary"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/meeting/transcript/{meeting_id}")
async def get_meeting_transcript(meeting_id: str, user=Depends(get_current_user)):
    """Fetch raw transcript"""
    try:
        # Implement database query logic
        return {"meeting_id": meeting_id, "transcript": "Meeting transcript"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Base (Q&A)
@app.post("/api/qa/query")
async def qa_query(qa_request: QARequest, user=Depends(get_current_user)):
    """Ask a question (LangChain RAG over KB)"""
    try:
        result = await workflow_graph.execute_workflow(
            "qa",
            qa_request.dict(),
            qa_request.user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa/upload")
async def qa_upload(file_data: Dict[str, Any], user=Depends(get_current_user)):
    """Upload new docs (PDF, TXT, etc.) to vector DB"""
    try:
        # Implement file processing and vector storage
        return {"message": "Document uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qa/search")
async def qa_search(q: str, user=Depends(get_current_user)):
    """Semantic search in KB"""
    try:
        # Implement vector search logic
        return {"query": q, "results": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Onboarding
@app.post("/api/onboarding/start")
async def start_onboarding(onboarding_request: OnboardingRequest, user=Depends(get_current_user)):
    """Start onboarding for a new hire"""
    try:
        result = await workflow_graph.execute_workflow(
            "onboarding",
            onboarding_request.dict(),
            onboarding_request.user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/status/{user_id}")
async def get_onboarding_status(user_id: str, user=Depends(get_current_user)):
    """Check onboarding task progress"""
    try:
        # Implement database query logic
        return {"user_id": user_id, "status": "in_progress", "tasks": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/onboarding/complete")
async def complete_onboarding_task(task_data: Dict[str, Any], user=Depends(get_current_user)):
    """Mark onboarding task as done"""
    try:
        # Implement task completion logic
        return {"message": "Task marked as complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Action Tracking & Tasks
@app.get("/api/tasks")
async def get_tasks(user=Depends(get_current_user)):
    """List all tasks (from standups/meetings/onboarding)"""
    try:
        # Implement database query logic
        return {"tasks": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks/update")
async def update_task(task_update: TaskUpdate, user=Depends(get_current_user)):
    """Update task status (done, blocked, in-progress)"""
    try:
        # Implement task update logic
        return {"message": "Task updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task_details(task_id: str, user=Depends(get_current_user)):
    """Fetch specific task details"""
    try:
        # Implement database query logic
        return {"task_id": task_id, "details": "Task details"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring / Observability
@app.get("/api/health")
async def health_check():
    """Health check for system uptime monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "autogen": "operational",
            "langgraph": "operational",
            "openai": "operational"
        }
    }

@app.get("/api/metrics")
async def get_metrics():
    """Prometheus metrics export"""
    return {"metrics": "prometheus_metrics_here"}

@app.get("/api/logs/{run_id}")
async def get_execution_logs(run_id: str, user=Depends(get_current_user)):
    """LangSmith/LangGraph execution trace"""
    try:
        # Implement logging retrieval logic
        return {"run_id": run_id, "logs": "Execution logs"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
