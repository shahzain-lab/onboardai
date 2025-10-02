import json
from datetime import datetime
from typing import Dict, Any, List

from config.env_config import config as env
from services.openai_agents import MCPToolManager, GeminiClient
from agents import Agent, Runner


class AgentsManager:
    """Direct specialist agents with MCP tools"""
    
    def __init__(self):
        self.gemini_api_key = env.GEMINI_API_KEY
        self.model = "gemini-2.0-flash-exp"
        
        self.gemini_client = GeminiClient(self.gemini_api_key, self.model)
        self.mcp_manager = MCPToolManager()
        self.runner = Runner()
        
        self.standup_agent = None
        self.qa_agent = None
        self.onboarding_agent = None
        self.summarizer_agent = None
        
        self.conversation_histories = {}
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        print("Initializing MCP servers...")
        await self.mcp_manager.initialize_mcp_servers()
        
        status = self.mcp_manager.get_connection_status()
        print(f"MCP Ready: {len(status['connected'])} servers, {status['total_tools']} tools")
        
        await self._setup_agents()
        self._initialized = True
        print("Agents ready!")
    
    async def _setup_agents(self):
        """Setup specialists with MCP tools"""
        
        db_tools = self.mcp_manager.get_tools_for_agent(["database", "knowledge_base"])
        all_tools = self.mcp_manager.get_tools_for_agent()
        print("all_tools", all_tools)
        
        # Standup Specialist
        self.standup_agent = Agent(
            name="standup_specialist",
            model=self.model,
            instructions="""Process standup requests directly.
You can directly call tools from the provided registry.  
Each tool is available as a Python function `.  

When a user asks something that requires database or knowledge base,  
you MUST call the appropriate tool and return its results  
 
Your tools:
- Database MCP: Store standup data

Steps:
1. Extract yesterday_tasks, today_tasks, blockers
2. Store in database
3. Return: "X completed, Y planned, Z blockers"

Be direct.""",
            tools=db_tools,
        )
        
        # QA Specialist
        self.qa_agent = Agent(
            name="qa_specialist",
            model=self.model,
            instructions="""Answer questions directly using available tools.
ou can directly call tools from the provided registry.  
Each tool is available as a Python function `.  

When a user asks something that requires database or knowledge base,  
you MUST call the appropriate tool and return its results  .

Your tools:
- Database MCP: Query tasks, users, data
- Knowledge Base MCP: Search documents (Pinecone vector DB), store new tasks, store explaination

For ANY user query:
1. Understand what they're asking
2. Use appropriate MCP tool to get data
3. Provide clear, direct answer

Examples:
- "hy" or "hi" → Greet user, ask how you can help
- "what is next task" → Query database for user's pending tasks
- "explain X" → Search knowledge base for documentation

Be helpful, direct, and use tools when needed.""",
            tools=all_tools,
        )
        
        # Onboarding Specialist
        self.onboarding_agent = Agent(
            name="onboarding_specialist",
            model=self.model,
            instructions="""Handle onboarding directly.
ou can directly call tools from the provided registry.  
Each tool is available as a Python function `.  

When a user asks something that requires database or knowledge base,  
you MUST call the appropriate tool and return its results.

Your tools:
- Database MCP: Create onboarding tasks
- Knowledge Base MCP: Get onboarding docs

Steps:
1. Create tasks in database
2. Return: "X tasks created"

Be direct.""",
            tools=all_tools,
        )
        
        # Summarizer
        self.summarizer_agent = Agent(
            name="summarizer",
            model=self.model,
            instructions="""ONE LINE summary (max 120 chars).

Extract final result only. No process details.

Examples:
- "Hello! How can I help you today?" → "Greeted user"
- "Your next task is..." → "Next task: [task name]"
- "3 tasks completed..." → "3 completed, 2 pending"

Be concise.""",
            mcp_servers=[],
        )
    
    async def _run_agent(self, agent: Agent, messages: List[Dict]) -> str:
        """Run agent with Gemini"""
        try:
            return await self.gemini_client.generate_response(messages, agent.mcp_servers)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def process_workflow(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow with direct specialist"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Select specialist
            specialists = {
                "standup": self.standup_agent,
                "qa": self.qa_agent,
                "onboarding": self.onboarding_agent,
                "meeting": self.qa_agent,
                "transcription": self.qa_agent,
            }
            
            specialist = specialists.get(workflow_type, self.qa_agent)
            
            print(f"→ {specialist.name} handling request")
            
            # Build message
            user_query = data.get("command_text", "") or json.dumps(data, indent=2)
            
            task_msg = f"""User query: {user_query}

User ID: {data.get('user_id', 'unknown')}

Process this request using your MCP tools and provide a direct answer."""
            
            conversation_id = f"{workflow_type}_{datetime.now().timestamp()}"
            messages = [{"role": "user", "content": task_msg}]
            
            # Call specialist
            response = await self._run_agent(specialist, messages)

            print("Response from Specialist => ", response)
            
            # Summarize
            summary_msg = [{"role": "user", "content": f"ONE LINE summary (max 120 chars):\n{response}"}]
            summary = await self._run_agent(self.summarizer_agent, summary_msg)
            
            # Store
            self.conversation_histories[conversation_id] = {
                "messages": messages + [{"role": "assistant", "content": response}],
                "agent": specialist.name,
                "workflow_type": workflow_type
            }
            
            return {
                "status": "success",
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


# Singleton
_agents_manager_instance = None

async def get_agents_manager() -> AgentsManager:
    global _agents_manager_instance
    if _agents_manager_instance is None:
        _agents_manager_instance = AgentsManager()
        await _agents_manager_instance.initialize()
    return _agents_manager_instance