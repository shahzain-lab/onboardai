
# load_dotenv(dotenv_path="../.env")

from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Annotated

# app = FastAPI()

class State(BaseModel):     
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def input_node_handler():
    return "input_node_handler"
def standup_node_handler():
    return "standup_node_handler"
def meeting_node_handler():
    return "meeting_node_handler"
def onboarding_node_handler():
    return "meeting_node_handler"
def qa_node_handler():
    return "meeting_node_handler"
def transcribe_node_handler():
    return "transcribe_node_handler"
def summarize_node_handler():
    return "summarize_node_handler"
def evaluator_node_handler():
    return "evaluator_node_handler"
def memory_node_handler():
    return "notify_node_handler"
# def recovery_node_handler():
#     return "recovery_node_handler"
# def notify_node_handler():
#     return "notify_node_handler"

def router_input_handler():
    return "notify_node_handler"
def router_onboarding_handler():
    return "notify_node_handler"


# Slack/Discord commands, Google Meet events, onboarding requests
graph_builder.add_node('input_node', input_node_handler)

# Brain - Decides which workflow should run
graph_builder.add_node('coordinator_node', input_node_handler)

# Collect updates from teams/users
graph_builder.add_node('standup_node', standup_node_handler)

# handle knowledge Q&A - Fetch from db + langchain RAG 
graph_builder.add_node('qa_node', qa_node_handler)

# Takes transcript - extract structured highlights + action items 
graph_builder.add_node('meeting_node', meeting_node_handler)

# onboard new hires - assign tasks, track progress
graph_builder.add_node('onboarding_node', onboarding_node_handler)

# convert meeting audio into text (whisper) - ensure accuracy
graph_builder.add_node('transcribe_node', transcribe_node_handler)

# summarize the standup updates and merge
graph_builder.add_node('summarize_node', summarize_node_handler)

# validate if output match the user intent. ensure clarity else handover to recovery
graph_builder.add_node('evaluator_node', evaluator_node_handler)

# long-term vector database storage
graph_builder.add_node('memory_node', memory_node_handler)

# create action plans - stores tasks, sync with db (may be Postresql)
graph_builder.add_node('action_tracker_node', memory_node_handler)
# graph_builder.add_node('recovery_node', recovery_node_handler)
# graph_builder.add_node('notify_node', notify_node_handler)

graph_builder.add_edge(START, "input_node")
graph_builder.add_edge("input_node", "coordinator_node")

graph_builder.add_conditional_edges("coordinator_node", router_input_handler, {
    "standup": "standup_node",
    "qa": "qa_node",
    "transcribe": "transcribe_node",
    "onboarding": "onboarding_node"
})

graph_builder.add_edge("standup_node", "summarize_node")
graph_builder.add_edge("summarize_node", "evaluator_node")
graph_builder.add_edge("evaluator_node", "action_tracker_node")

graph_builder.add_edge("transcribe_node", "meeting_node")
graph_builder.add_edge("meeting_node", "evaluator_node")
graph_builder.add_edge("evaluator_node", "action_tracker_node")

graph_builder.add_edge("qa_node", "evaluator_node")

graph_builder.add_conditional_edges("onboarding_node", router_onboarding_handler, {
    "action": "action_tracker_node",
    "memory": "memory_node"
})


gragh = graph_builder.compile()

with open("graph.png", "wb") as f:
    f.write(gragh.get_graph().draw_mermaid_png())


# if __name__ == "__main__":
#     app()
