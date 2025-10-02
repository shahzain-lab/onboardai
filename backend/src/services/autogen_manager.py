import json
from datetime import datetime
from typing import Dict, Any, List

from config.env_config import config as env
from services.openai_agents import MCPToolManager, GeminiClient
from agents import Agent, Runner


class AgentsManager:
    """Simplified agents manager - direct specialist calls only"""
    
    def __init__(self):
        self.gemini_api_key = env.GEMINI_API_KEY
        self.model = "gemini-2.0-flash-exp"
        
        self.gemini_client = GeminiClient(self.gemini_api_key, self.model)
        self.mcp_manager = MCPToolManager()
        self.runner = Runner()
        
        # Only specialist agents
        self.standup_agent = None
        self.qa_agent = None
        self.onboarding_agent = None
        self.summarizer_agent = None
        
        self.conversation_histories = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize MCP servers and agents"""
        if self._initialized:
            return
        
        print("Initializing MCP servers...")
        await self.mcp_manager.initialize_mcp_servers()
        
        status = self.mcp_manager.get_connection_status()
        print(f"MCP: {len(status['connected'])} servers, {status['total_tools']} tools")
        
        if status['failed']:
            print(f"Warning: {status['failed']}")
        
        print("Creating specialist agents...")
        await self._setup_agents()
        
        self._initialized = True
        print("System ready!")
    
    async def _setup_agents(self):
        """Setup only specialist agents - no coordinator, no meeting agent"""
        
        db_tools = self.mcp_manager.get_tools_for_agent(["database"])
        all_tools = self.mcp_manager.get_tools_for_agent()
        
        # 1. Standup Specialist
        self.standup_agent = Agent(
            name="standup_specialist",
            model=self.model,
            instructions="""Process standup data directly.

            Steps:
            1. Parse yesterday_tasks, today_tasks, blockers from input
            2. Store in database using MCP tools
            3. Return brief summary

            Output format: "X completed, Y planned, Z blockers"
            Be direct and concise.""",
            mcp_servers=db_tools,
        )
        
        # 2. QA Specialist
        self.qa_agent = Agent(
            name="qa_specialist",
            model=self.model,
            instructions="""Answer questions and handle task queries directly.

            Steps:
            1. Understand the question
            2. Query database for tasks using user_id
            3. Return clear answer

            For "what is next task":
            - Query tasks table WHERE user_id = <user_id> AND status IN ('pending', 'in_progress')
            - Return the highest priority task

            Output format: "Next task: [task name]"
            Be direct, helpful, concise.""",
            mcp_servers=all_tools,
        )
        
        # 3. Onboarding Specialist
        self.onboarding_agent = Agent(
            name="onboarding_specialist",
            model=self.model,
            instructions="""Handle onboarding workflows directly.

            Steps:
            1. Create onboarding tasks in database
            2. Store onboarding plan
            3. Return summary

            Output format: "X tasks created for onboarding"
            Be direct and concise.""",
            mcp_servers=all_tools,
        )
        
        # 4. Summarizer
        self.summarizer_agent = Agent(
            name="summarizer",
            model=self.model,
            instructions="""Create ONE LINE summary (max 120 chars).

            Rules:
            - Extract final result only
            - Remove all process details
            - No technical jargon
            - Plain language
            - Present tense

            Example: "3 tasks completed, 2 blockers identified" """,
            mcp_servers=[],
        )
    
    async def _run_agent_with_gemini(self, agent: Agent, messages: List[Dict]) -> str:
        """Run agent with Gemini"""
        try:
            return await self.gemini_client.generate_response(messages, agent.mcp_servers)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def process_workflow(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow - direct specialist call (no coordinator)
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Direct specialist selection
            specialist_map = {
                "standup": self.standup_agent,
                "qa": self.qa_agent,
                "onboarding": self.onboarding_agent,
                "meeting": self.qa_agent,  # Meeting queries go to QA
                "transcription": self.qa_agent,  # Transcription queries go to QA
            }
            
            specialist = specialist_map.get(workflow_type, self.qa_agent)
            
            print(f"→ Calling {specialist.name} directly")
            
            # Build message
            task_msg = f"""Handle this {workflow_type} request:

{json.dumps(data, indent=2)}

Process using your MCP tools and return the result."""
            
            conversation_id = f"{workflow_type}_{datetime.now().timestamp()}"
            messages = [{"role": "user", "content": task_msg}]
            
            # Call specialist
            response = await self._run_agent_with_gemini(specialist, messages)
            
            # Summarize
            summary_msg = [{"role": "user", "content": f"Summarize in ONE LINE (max 120 chars):\n\n{response}"}]
            summary = await self._run_agent_with_gemini(self.summarizer_agent, summary_msg)
            
            # Store conversation
            self.conversation_histories[conversation_id] = {
                "messages": messages + [{"role": "assistant", "content": response}],
                "agent": specialist.name,
                "workflow_type": workflow_type
            }
            
            # Check for MCP errors
            has_error = "error" in response.lower() or "failed" in response.lower()
            
            return {
                "status": "success" if not has_error else "partial_success",
                "workflow_type": workflow_type,
                "result": response,
                "summary": summary.strip(),
                "agent_used": specialist.name,
                "conversation_id": conversation_id,
                "mcp_status": self.mcp_manager.get_connection_status(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "workflow_type": workflow_type,
                "error": str(e),
                "mcp_status": self.mcp_manager.get_connection_status(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_simple_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process simple query with QA agent"""
        if not self._initialized:
            await self.initialize()
        
        try:
            context_str = f"\nContext: {json.dumps(context)}" if context else ""
            messages = [{"role": "user", "content": f"{query}{context_str}"}]
            return await self._run_agent_with_gemini(self.qa_agent, messages)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def continue_conversation(self, conversation_id: str, message: str) -> str:
        """Continue conversation"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if conversation_id not in self.conversation_histories:
                return "Conversation not found"
            
            history = self.conversation_histories[conversation_id]
            agent_name = history["agent"]
            
            agent_map = {
                "standup_specialist": self.standup_agent,
                "qa_specialist": self.qa_agent,
                "onboarding_specialist": self.onboarding_agent,
            }
            agent = agent_map.get(agent_name, self.qa_agent)
            
            history["messages"].append({"role": "user", "content": message})
            response = await self._run_agent_with_gemini(agent, history["messages"])
            history["messages"].append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"


# Singleton
_agents_manager_instance = None

async def get_agents_manager() -> AgentsManager:
    """Get or create agents manager singleton"""
    global _agents_manager_instance
    if _agents_manager_instance is None:
        _agents_manager_instance = AgentsManager()
        await _agents_manager_instance.initialize()
    return _agents_manager_instance