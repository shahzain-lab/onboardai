# ============================================================================
# PYDANTIC MODELS
# ============================================================================
from langgraph.graph.message import add_messages
from typing import Dict, Any, List, Annotated, Optional
from pydantic import BaseModel

class State(BaseModel):
    messages: Annotated[list, add_messages]
    workflow_type: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SlackEventRequest(BaseModel):
    token: str
    team_id: str
    api_app_id: str
    event: Dict[str, Any]
    type: str
    event_id: str
    event_time: int

class SlackCommandRequest(BaseModel):
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str
    response_url: str

class StandupRequest(BaseModel):
    user_id: str
    yesterday_tasks: List[str]
    today_tasks: List[str]
    blockers: List[str]

class MeetingRequest(BaseModel):
    meeting_id: str
    recording_url: Optional[str] = None
    transcript: Optional[str] = None
    participants: List[str] = []

class QARequest(BaseModel):
    question: str
    context: Optional[str] = None
    user_id: str

class OnboardingRequest(BaseModel):
    user_id: str
    role: str
    start_date: str
    manager_id: str

class TaskUpdate(BaseModel):
    task_id: str
    status: str  # "done", "blocked", "in-progress"
    notes: Optional[str] = None
