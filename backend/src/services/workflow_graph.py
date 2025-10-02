from services.autogen_manager import get_agents_manager, AgentsManager
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from config.pydantic_models import State
from typing import Dict, Any


class WorkflowGraph:
    """Direct specialist routing - NO coordinator node"""
    
    def __init__(self, agents_manager: AgentsManager = None):
        self.agents_manager = agents_manager
        self.graph = None
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        if self.agents_manager is None:
            self.agents_manager = await get_agents_manager()
        
        self.graph = self.build_graph()
        self._initialized = True
    
    def build_graph(self):
        """Build graph with direct specialist routing"""
        graph_builder = StateGraph(State)
        
        # ====================================================================
        # SINGLE NODE HANDLERS - Direct specialist processing
        # ====================================================================
        
        async def standup_handler(state: State) -> State:
            """Direct standup specialist call"""
            try:
                result = await self.agents_manager.process_workflow("standup", state.metadata)
                state.metadata.update({
                    "workflow_result": result,
                    "final_summary": result.get("summary", "Standup completed"),
                    "conversation_id": result.get("conversation_id"),
                    "mcp_status": result.get("mcp_status", {})
                })
            except Exception as e:
                state.metadata.update({
                    "workflow_result": {"status": "error", "error": str(e)},
                    "final_summary": f"Error: {str(e)}"
                })
            return state
        
        async def qa_handler(state: State) -> State:
            """Direct QA specialist call"""
            try:
                result = await self.agents_manager.process_workflow("qa", state.metadata)
                state.metadata.update({
                    "workflow_result": result,
                    "final_summary": result.get("summary", "Query processed"),
                    "conversation_id": result.get("conversation_id"),
                    "mcp_status": result.get("mcp_status", {})
                })
            except Exception as e:
                state.metadata.update({
                    "workflow_result": {"status": "error", "error": str(e)},
                    "final_summary": f"Error: {str(e)}"
                })
            return state
        
        async def onboarding_handler(state: State) -> State:
            """Direct onboarding specialist call"""
            try:
                result = await self.agents_manager.process_workflow("onboarding", state.metadata)
                state.metadata.update({
                    "workflow_result": result,
                    "final_summary": result.get("summary", "Onboarding started"),
                    "conversation_id": result.get("conversation_id"),
                    "mcp_status": result.get("mcp_status", {})
                })
            except Exception as e:
                state.metadata.update({
                    "workflow_result": {"status": "error", "error": str(e)},
                    "final_summary": f"Error: {str(e)}"
                })
            return state
        
        # ====================================================================
        # ROUTER - Direct to specialist based on workflow type
        # ====================================================================
        
        def route_to_specialist(state: State) -> str:
            """Direct routing to specialist"""
            routes = {
                "standup": "standup",
                "qa": "qa",
                "onboarding": "onboarding",
                "meeting": "qa",  # meetings handled by QA
                "transcription": "qa"  # transcriptions handled by QA
            }
            return routes.get(state.workflow_type, "qa")
        
        # ====================================================================
        # BUILD GRAPH - Simple direct routing
        # ====================================================================
        
        graph_builder.add_node('standup', standup_handler)
        graph_builder.add_node('qa', qa_handler)
        graph_builder.add_node('onboarding', onboarding_handler)
        
        # Direct routing from START
        graph_builder.add_conditional_edges(
            START,
            route_to_specialist,
            {
                "standup": "standup",
                "qa": "qa",
                "onboarding": "onboarding"
            }
        )
        
        # All nodes go to END
        graph_builder.add_edge("standup", END)
        graph_builder.add_edge("qa", END)
        graph_builder.add_edge("onboarding", END)
        
        checkpointer = InMemorySaver()
        return graph_builder.compile(checkpointer=checkpointer)
    
    async def execute_workflow(
        self,
        workflow_type: str,
        data: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute workflow"""
        if not self._initialized:
            await self.initialize()
        
        print(f"\n{'='*60}")
        print(f"Workflow: {workflow_type} | User: {user_id}")
        print(f"{'='*60}")
        
        state = State(
            messages=[],
            workflow_type=workflow_type,
            user_id=user_id,
            metadata=data
        )
        
        try:
            result = await self.graph.ainvoke(state, config={"configurable": {"thread_id": "1"}})
            
            metadata = result.get("metadata", {})
            workflow_result = metadata.get("workflow_result", {})
            
            print(f"Summary: {metadata.get('final_summary', 'N/A')}")
            print(f"{'='*60}\n")
            
            return {
                "status": "success",
                "workflow_type": workflow_type,
                "user_id": user_id,
                "summary": metadata.get("final_summary", "Completed"),
                "full_result": workflow_result.get("result"),
                "agent_used": workflow_result.get("agent_used"),
                "conversation_id": metadata.get("conversation_id"),
                "mcp_status": metadata.get("mcp_status", {})
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"{'='*60}\n")
            
            return {
                "status": "error",
                "workflow_type": workflow_type,
                "user_id": user_id,
                "summary": f"Error: {str(e)}",
                "error": str(e)
            }

workflow_graph = WorkflowGraph()