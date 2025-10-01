# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================
from services.autogen_manager import get_agents_manager, AgentsManager
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from config.pydantic_models import State
from typing import Dict, Any


class WorkflowGraph:
    """
    LangGraph-based workflow orchestrator that uses AgentsManager for execution.
    
    This provides:
    - High-level workflow routing
    - State management across nodes
    - Conditional routing based on workflow type
    - Checkpointing and memory
    """
    
    def __init__(self, agents_manager: AgentsManager = None):
        self.agents_manager = agents_manager
        self.graph = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize agents manager and build graph"""
        if self._initialized:
            return
        
        if self.agents_manager is None:
            self.agents_manager = await get_agents_manager()
        
        self.graph = self.build_graph()
        self._initialized = True
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        graph_builder = StateGraph(State)
        
        # ====================================================================
        # NODE HANDLERS
        # ====================================================================
        
        async def input_node_handler(state: State) -> State:
            """Process initial input and validate workflow type"""
            print(f"Input Node: Processing {state.workflow_type} workflow")
            
            valid_workflows = ["standup", "meeting", "qa", "onboarding", "transcription"]
            if state.workflow_type not in valid_workflows:
                print(f"Unknown workflow type: {state.workflow_type}, defaulting to 'qa'")
                state.workflow_type = "qa"
            
            state.metadata["workflow_started_at"] = state.metadata.get("timestamp", "")
            return state
        
        async def coordinator_node_handler(state: State) -> State:
            """Coordinator node that analyzes and routes requests"""
            print(f"Coordinator Node: Analyzing {state.workflow_type} workflow")
            
            workflow_type = state.workflow_type or "general"
            
            if workflow_type == "general" or not state.metadata:
                result = await self.agents_manager.process_simple_query(
                    query="Process this general request",
                    context=state.metadata
                )
                state.metadata.update({
                    "coordinator_result": {
                        "status": "success",
                        "result": result,
                        "handled_by": "coordinator"
                    }
                })
            else:
                state.metadata.update({
                    "coordinator_analysis": "Routing to specialist agent",
                    "ready_for_specialist": True
                })
            
            return state
        
        async def standup_node_handler(state: State) -> State:
            """Handle standup workflow"""
            print(f"Standup Node: Processing standup for user {state.user_id}")
            
            try:
                result = await self.agents_manager.process_workflow("standup", state.metadata)
                print(f"Standup Result: {result.get('status')}")
                
                state.metadata.update({
                    "standup_result": result,
                    "workflow_completed": result.get("status") == "success"
                })
                
                if "conversation_id" in result:
                    state.metadata["conversation_id"] = result["conversation_id"]
                
            except Exception as e:
                print(f"Standup Node Error: {e}")
                state.metadata.update({
                    "standup_result": {"status": "error", "error": str(e)},
                    "workflow_completed": False
                })
            
            return state
        
        async def meeting_node_handler(state: State) -> State:
            """Handle meeting processing workflow"""
            print(f"Meeting Node: Processing meeting {state.metadata.get('meeting_id', 'unknown')}")
            
            try:
                result = await self.agents_manager.process_workflow("meeting", state.metadata)
                print(f"Meeting Result: {result.get('status')}")
                
                state.metadata.update({
                    "meeting_result": result,
                    "workflow_completed": result.get("status") == "success"
                })
                
                if "conversation_id" in result:
                    state.metadata["conversation_id"] = result["conversation_id"]
                
            except Exception as e:
                print(f"Meeting Node Error: {e}")
                state.metadata.update({
                    "meeting_result": {"status": "error", "error": str(e)},
                    "workflow_completed": False
                })
            
            return state
        
        async def qa_node_handler(state: State) -> State:
            """Handle Q&A workflow"""
            print(f"QA Node: Processing question from user {state.user_id}")
            
            try:
                result = await self.agents_manager.process_workflow("qa", state.metadata)
                print(f"QA Result: {result.get('status')}")
                
                state.metadata.update({
                    "qa_result": result,
                    "workflow_completed": result.get("status") == "success"
                })
                
                if "conversation_id" in result:
                    state.metadata["conversation_id"] = result["conversation_id"]
                
            except Exception as e:
                print(f"QA Node Error: {e}")
                state.metadata.update({
                    "qa_result": {"status": "error", "error": str(e)},
                    "workflow_completed": False
                })
            
            return state
        
        async def onboarding_node_handler(state: State) -> State:
            """Handle onboarding workflow"""
            print(f"Onboarding Node: Onboarding user {state.metadata.get('user_id', 'unknown')}")
            
            try:
                result = await self.agents_manager.process_workflow("onboarding", state.metadata)
                print(f"Onboarding Result: {result.get('status')}")
                
                state.metadata.update({
                    "onboarding_result": result,
                    "workflow_completed": result.get("status") == "success"
                })
                
                if "conversation_id" in result:
                    state.metadata["conversation_id"] = result["conversation_id"]
                
            except Exception as e:
                print(f"Onboarding Node Error: {e}")
                state.metadata.update({
                    "onboarding_result": {"status": "error", "error": str(e)},
                    "workflow_completed": False
                })
            
            return state
        
        async def transcribe_node_handler(state: State) -> State:
            """Handle transcription workflow"""
            print(f"Transcribe Node: Transcribing recording {state.metadata.get('recording_url', 'unknown')}")
            
            try:
                if "transcript" not in state.metadata:
                    state.metadata["transcript"] = "[Transcription placeholder]"
                    state.metadata["transcription_status"] = "placeholder"
                else:
                    state.metadata["transcription_status"] = "provided"
                
                state.metadata.update({
                    "transcription_result": {
                        "status": "success",
                        "message": "Transcript ready for processing"
                    },
                    "ready_for_meeting_node": True
                })
                
                print("Transcription completed")
                
            except Exception as e:
                print(f"Transcribe Node Error: {e}")
                state.metadata.update({
                    "transcription_result": {"status": "error", "error": str(e)},
                    "ready_for_meeting_node": False
                })
            
            return state
        
        async def summarize_node_handler(state: State) -> State:
            """Generate final 1-line summary of the workflow result"""
            print(f"Summarize Node: Creating concise summary")
            
            try:
                # Get the workflow result from metadata
                workflow_type = state.workflow_type
                result_key = f"{workflow_type}_result"
                
                if result_key in state.metadata:
                    result_data = state.metadata[result_key]
                    
                    # Extract the full response
                    full_response = result_data.get("result", "")
                    
                    # If we already have a summary from agents_manager, use it
                    if "summary" in result_data:
                        summary = result_data["summary"]
                    else:
                        # Generate summary using summarizer agent
                        summary_messages = [
                            {"role": "user", "content": f"Summarize this in ONE LINE (max 150 chars):\n\n{full_response}"}
                        ]
                        summary = await self.agents_manager._run_agent_with_gemini(
                            self.agents_manager.summarizer_agent, 
                            summary_messages
                        )
                        summary = summary.strip()
                    
                    # Add summary to metadata
                    state.metadata["final_summary"] = summary
                    state.metadata["summary_generated"] = True
                    
                    print(f"Summary: {summary}")
                else:
                    state.metadata["final_summary"] = f"{workflow_type} workflow completed"
                    state.metadata["summary_generated"] = False
                
            except Exception as e:
                print(f"Summarize Node Error: {e}")
                state.metadata["final_summary"] = f"{state.workflow_type} workflow completed with errors"
                state.metadata["summary_generated"] = False
            
            return state
        
        # ====================================================================
        # ROUTER FUNCTIONS
        # ====================================================================
        
        def router_input_handler(state: State) -> str:
            """Route from coordinator to appropriate specialist node"""
            workflow_type = state.workflow_type or "general"
            
            workflow_routes = {
                "standup": "standup",
                "qa": "qa",
                "transcription": "transcribe",
                "onboarding": "onboarding",
                "meeting": "meeting"
            }
            
            route = workflow_routes.get(workflow_type, "qa")
            print(f"Router: Routing {workflow_type} -> {route}_node")
            
            return route
        
        # ====================================================================
        # BUILD GRAPH STRUCTURE
        # ====================================================================
        
        # Add all nodes
        graph_builder.add_node('input_node', input_node_handler)
        graph_builder.add_node('coordinator_node', coordinator_node_handler)
        graph_builder.add_node('standup_node', standup_node_handler)
        graph_builder.add_node('qa_node', qa_node_handler)
        graph_builder.add_node('meeting_node', meeting_node_handler)
        graph_builder.add_node('onboarding_node', onboarding_node_handler)
        graph_builder.add_node('transcribe_node', transcribe_node_handler)
        graph_builder.add_node('summarize_node', summarize_node_handler)  # Add summarizer node
        
        # Define edges
        graph_builder.add_edge(START, "input_node")
        graph_builder.add_edge("input_node", "coordinator_node")
        
        graph_builder.add_conditional_edges(
            "coordinator_node",
            router_input_handler,
            {
                "standup": "standup_node",
                "qa": "qa_node",
                "transcribe": "transcribe_node",
                "onboarding": "onboarding_node",
                "meeting": "meeting_node"
            }
        )
        
        # Terminal nodes - all flow through summarizer before END
        graph_builder.add_edge("standup_node", "summarize_node")
        graph_builder.add_edge("qa_node", "summarize_node")
        graph_builder.add_edge("onboarding_node", "summarize_node")
        graph_builder.add_edge("meeting_node", "summarize_node")
        
        # Transcription flows to meeting, then to summarizer
        graph_builder.add_edge("transcribe_node", "meeting_node")
        
        # Summarizer is the final node before END
        graph_builder.add_edge("summarize_node", END)
        
        # Compile graph with memory checkpointing
        checkpointer = InMemorySaver()
        compiled_graph = graph_builder.compile(checkpointer=checkpointer)
        
        return compiled_graph
    
    async def execute_workflow(
        self,
        workflow_type: str,
        data: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute a workflow through the LangGraph"""
        if not self._initialized:
            await self.initialize()
        
        print(f"\n{'='*60}")
        print(f"Starting Workflow: {workflow_type}")
        print(f"User: {user_id or 'anonymous'}")
        print(f"{'='*60}\n")
        
        # Create initial state
        state = State(
            messages=[],
            workflow_type=workflow_type,
            user_id=user_id,
            metadata=data
        )
        
        try:
            # Execute the graph
            result = await self.graph.ainvoke(state, config= {"configurable": {"thread_id": "1"}})
            
            print(f"\n{'='*60}")
            print(f"Workflow Completed: {workflow_type}")
            print(f"Workflow Result: {result}")
            print(f"{'='*60}\n")
            
            return {
                "status": "success",
                "workflow_type": workflow_type,
                "user_id": user_id,
                "result": result.get("metadata"),
                "workflow_completed": result.get("metadata").get("workflow_completed", False),
                "conversation_id": result.get("metadata").get("conversation_id"),
                "architecture": "LangGraph → AgentsManager → Gemini + MCP"
            }
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Workflow Failed: {workflow_type}")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            
            return {
                "status": "error",
                "workflow_type": workflow_type,
                "user_id": user_id,
                "error": str(e),
                "architecture": "LangGraph → AgentsManager → Gemini + MCP"
            }
        
workflow_graph = WorkflowGraph()