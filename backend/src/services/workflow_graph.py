# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================
from services.autogen_manager import autogen_manager, AutoGenManager
from langgraph.graph import StateGraph, START, END
from config.pydantic_models import State
from typing import Dict, Any


class WorkflowGraph:
    def __init__(self, autogen_manager: AutoGenManager):
        self.autogen_manager = autogen_manager
        self.graph = self.build_graph()
    
    def build_graph(self):
        graph_builder = StateGraph(State)
        
        # Node handlers
        async def input_node_handler(state: State) -> State:
            """Process initial input and determine workflow type"""
            # Basic input validation and routing logic
            return state
        
        async def coordinator_node_handler(state: State) -> State:
            """Main coordination logic"""
            workflow_type = state.workflow_type or "general"
            result = await self.autogen_manager.process_workflow(
                workflow_type, 
                state.metadata
            )
            state.metadata.update({"coordinator_result": result})
            return state
        
        async def standup_node_handler(state: State) -> State:
            """Handle standup workflow"""
            result = await self.autogen_manager.process_workflow("standup", state.metadata)
            state.metadata.update({"standup_result": result})
            return state
        
        async def meeting_node_handler(state: State) -> State:
            """Handle meeting processing workflow"""
            result = await self.autogen_manager.process_workflow("meeting", state.metadata)
            state.metadata.update({"meeting_result": result})
            return state
        
        async def qa_node_handler(state: State) -> State:
            """Handle Q&A workflow"""
            result = await self.autogen_manager.process_workflow("qa", state.metadata)
            state.metadata.update({"qa_result": result})
            return state
        
        async def onboarding_node_handler(state: State) -> State:
            """Handle onboarding workflow"""
            result = await self.autogen_manager.process_workflow("onboarding", state.metadata)
            state.metadata.update({"onboarding_result": result})
            return state
        
        async def transcribe_node_handler(state: State) -> State:
            """Handle transcription workflow"""
            result = await self.autogen_manager.process_workflow("transcription", state.metadata)
            state.metadata.update({"transcription_result": result})
            return state
        
        # Router functions
        def router_input_handler(state: State) -> str:
            """Route based on workflow type"""
            workflow_type = state.workflow_type or "general"
            workflow_routes = {
                "standup": "standup",
                "qa": "qa", 
                "transcription": "transcribe",
                "onboarding": "onboarding",
                "meeting": "meeting"
            }
            return workflow_routes.get(workflow_type, "qa")
        
        def router_onboarding_handler(state: State) -> str:
            """Route onboarding to appropriate next step"""
            return "action"
        
        # Add nodes
        graph_builder.add_node('input_node', input_node_handler)
        graph_builder.add_node('coordinator_node', coordinator_node_handler)
        graph_builder.add_node('standup_node', standup_node_handler)
        graph_builder.add_node('qa_node', qa_node_handler)
        graph_builder.add_node('meeting_node', meeting_node_handler)
        graph_builder.add_node('onboarding_node', onboarding_node_handler)
        graph_builder.add_node('transcribe_node', transcribe_node_handler)
        
        # Add edges
        graph_builder.add_edge(START, "input_node")
        graph_builder.add_edge("input_node", "coordinator_node")
        
        graph_builder.add_conditional_edges("coordinator_node", router_input_handler, {
            "standup": "standup_node",
            "qa": "qa_node",
            "transcribe": "transcribe_node", 
            "onboarding": "onboarding_node",
            "meeting": "meeting_node"
        })
        
        graph_builder.add_edge("standup_node", END)
        graph_builder.add_edge("transcribe_node", "meeting_node")
        graph_builder.add_edge("meeting_node", END)
        graph_builder.add_edge("qa_node", END)
        graph_builder.add_edge("onboarding_node", END)
        
        return graph_builder.compile()
    
    async def execute_workflow(self, workflow_type: str, data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Execute a workflow through the graph"""
        state = State(
            messages=[],
            workflow_type=workflow_type,
            user_id=user_id,
            metadata=data
        )
        
        result = await self.graph.ainvoke(state)
        return result


workflow_graph = WorkflowGraph(autogen_manager)
