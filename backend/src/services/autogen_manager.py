import json
import os
from datetime import datetime
from typing import Dict, Any, List

from config.env_config import config as env
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI

class AgentsManager:
    """Direct specialist agents with MCP tools using OpenAI"""
    
    def __init__(self):
        self.model = "gpt-4o-mini"  # Using GPT-4o mini model
        self.runner = Runner()
        
        self.standup_agent = None
        self.qa_agent = None
        self.onboarding_agent = None
        self.summarizer_agent = None
        self.mcp_servers = None
        self._mcp_server_contexts = []  # Store context managers
        
        self.conversation_histories = {}
        self._initialized = False

        # 1. Set the correct base URL for the Gemini API's OpenAI compatibility layer
        BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        GEMINI_API_KEY = env.GEMINI_API_KEY # Ensure this is set in your environment

        # 2. Initialize the OpenAI-compatible client
        gemini_client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key=GEMINI_API_KEY,
        )

        # 3. Create the model instance using the Agents SDK class and the custom client
        self.gemini_modal = OpenAIChatCompletionsModel(
            model="gemini-2.5-pro", # Use the specific Gemini model name
            openai_client=gemini_client,
        )
    
    async def initialize(self):
        """Initialize MCP servers and agents"""
        if self._initialized:
            return
        
        print("Initializing MCP servers...")
        
        # Get absolute paths to MCP server files
        db_server_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "servers", "database_tools.py")
        )
        kb_server_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "servers", "kb_vector_tools.py")
        )
        
        # Verify server files exist
        if not os.path.exists(db_server_path):
            raise FileNotFoundError(f"Database server not found: {db_server_path}")
        if not os.path.exists(kb_server_path):
            raise FileNotFoundError(f"Knowledge base server not found: {kb_server_path}")
        
        # Create MCP server instances
        db_server = MCPServerStdio(
            params={
                "command": "python",
                "args": [db_server_path],
                "env": {
                    "DATABASE_URL": env.DATABASE_URL
                }
            }
        )
        
        kb_server = MCPServerStdio(
            params={
                "command": "python",
                "args": [kb_server_path],
                "env": {
                    "PINECONE_API_KEY": env.PINECONE_API_KEY
                }
            }
        )
        
        # Connect to MCP servers using async context manager
        # Enter the context managers and keep them alive
        await db_server.__aenter__()
        await kb_server.__aenter__()
        
        # Store the servers and context managers
        self.mcp_servers = [db_server, kb_server]
        self._mcp_server_contexts = [db_server, kb_server]
        
        print(f"✓ Connected to {len(self.mcp_servers)} MCP servers")
        
        await self._setup_agents()
        self._initialized = True
        print("Agents ready!")
    
    async def _setup_agents(self):
        """Setup specialist agents with MCP tools"""
        
        # Standup Specialist
        self.standup_agent = Agent(
            name="standup_specialist",
            model=self.gemini_modal,
            instructions="""Process standup requests directly.

You have access to database and knowledge base tools through MCP servers.

**Your Tools:**
- Database tools for storing standup data (create_task, update_task, etc.)
- Knowledge base tools for retrieving information

**Process:**
1. Extract yesterday_tasks, today_tasks, and blockers from the user's input
2. Store the standup data in the database using the appropriate tools
3. Return a clear summary: "X tasks completed yesterday, Y tasks planned today, Z blockers"

**Important:**
- ALWAYS use the database tools to store data
- Don't make up information - use the tools to get/store real data
- Be direct and concise in your responses

Be helpful and efficient.""",
            mcp_servers=self.mcp_servers,
        )
        
        # QA Specialist
        self.qa_agent = Agent(
            name="qa_specialist",
            model=self.gemini_modal,
            instructions="""Answer questions directly using available tools.

**Database Tools (PostgreSQL):**
- list_tasks: List tasks with optional filters (user_id, status, limit)
  Example: list_tasks(status="pending", limit=10)
  
- get_task: Get specific task by ID
  Example: get_task(task_id=123)
  
- create_task: Create a new task
  Example: create_task(user_id="user123", title="New Task", description="Details", status="pending", priority="high", due_date="2024-10-15")
  
- update_task: Update task fields
  Example: update_task(task_id=123, status="completed", completed_at="2024-10-03")
  
- get_user: Get user details by ID
  Example: get_user(user_id="user123")
  
- list_users: List recent users
  Example: list_users(limit=20)
  
- create_user: Create or update a user
  Example: create_user(user_id="user123", email="user@example.com", name="John Doe", role="admin")
  
- raw_read: Execute raw SQL SELECT queries (use carefully)
  Example: raw_read(sql="SELECT * FROM tasks WHERE status = 'pending'", limit=50)

**Knowledge Base Tools (Pinecone Vector DB):**
- kb_query: Search knowledge base with text
  Example: kb_query(query_text="What is machine learning?", top_k=5, namespace="docs")
  
- kb_answer_qa: Answer questions using KB context
  Example: kb_answer_qa(question="How does AI work?", top_k=3, context_key="text")
  
- kb_upsert_text: Insert text documents with auto-embedding
  Example: kb_upsert_text(documents=[{"id": "doc1", "text": "Content...", "metadata": {"category": "tech"}}], namespace="docs")
  
- kb_stats: Get index statistics
  Example: kb_stats(namespace="docs")
  
- kb_delete: Delete vectors from index
  Example: kb_delete(ids=["doc1", "doc2"], namespace="docs")

**Critical Instructions:**
1. **ALWAYS use the tools** to get real data - never make up information
2. For task queries, use list_tasks with appropriate filters
3. For user queries, use get_user or list_users
4. For knowledge base queries, use kb_query or kb_answer_qa
5. Parse tool results carefully and present them clearly
6. If a tool returns an error, explain it to the user and suggest alternatives
7. When creating or updating records, confirm the action was successful

Be helpful, direct, and always use tools when needed.""",
            mcp_servers=self.mcp_servers,
        )
        
        # Onboarding Specialist
        self.onboarding_agent = Agent(
            name="onboarding_specialist",
            model=self.gemini_modal,
            instructions="""Handle onboarding requests directly.

You have access to database and knowledge base tools through MCP servers.

**Your Tools:**
- Database tools: Create and manage onboarding tasks
- Knowledge base tools: Retrieve onboarding documentation and resources

**Process:**
1. Use the database tools to create onboarding tasks for the new user
2. Use knowledge base tools to get relevant onboarding documentation if needed
3. Return a clear summary: "Created X onboarding tasks for [user]"

**Important:**
- ALWAYS use the tools to create tasks and retrieve information
- Don't make up information - use the tools
- Be direct and helpful

Be efficient and welcoming.""",
            mcp_servers=self.mcp_servers,
        )
        
        # Summarizer Agent
        self.summarizer_agent = Agent(
            name="summarizer",
            model=self.gemini_modal,
            instructions="""Create ONE LINE summaries (max 120 characters).

Extract the final result only. No process details.

**Examples:**
- "Hello! How can I help you today?" → "Greeted user"
- "Your next task is to review the Q4 report..." → "Next task: Review Q4 report"
- "3 tasks completed yesterday, 2 tasks planned for today" → "3 completed, 2 planned"
- "Created 5 onboarding tasks for new user" → "5 tasks created"

**Rules:**
- Maximum 120 characters
- No filler words
- Focus on the action or key information
- Be concise and clear

Be extremely concise.""",
            mcp_servers=self.mcp_servers
        )
    
    async def _run_agent(self, agent: Agent, input_message: str) -> str:
        """Run agent with OpenAI using Runner"""
        try:
            # Use Runner.run to execute the agent
            result = await Runner.run(
                starting_agent=agent,
                input=input_message
            )
            
            # Return the final output from the agent
            return result.final_output
            
        except Exception as e:
            print(f"Error running agent {agent.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    async def process_workflow(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow with direct specialist"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Select specialist based on workflow type
            specialists = {
                "standup": self.standup_agent,
                "qa": self.qa_agent,
                "ask": self.qa_agent,
                "onboarding": self.onboarding_agent,
                "meeting": self.qa_agent,
                "transcription": self.qa_agent,
            }
            
            specialist = specialists.get(workflow_type, self.qa_agent)
            
            print(f"\n{'='*60}")
            print(f"Workflow: {workflow_type} | User: {data.get('user_id', 'unknown')}")
            print(f"{'='*60}")
            print(f"→ {specialist.name} handling request")
            
            # Build user query message
            user_query = data.get("command_text", "") or json.dumps(data, indent=2)
            
            task_msg = f"""User query: {user_query}

User ID: {data.get('user_id', 'unknown')}

Process this request using your MCP tools and provide a direct answer."""
            
            # Run the specialist agent
            response = await self._run_agent(specialist, task_msg)
            
            print(f"Response from Specialist => {response}")
            
            # Generate summary using summarizer agent
            summary_msg = f"ONE LINE summary (max 120 chars):\n{response}"
            summary = await self._run_agent(self.summarizer_agent, summary_msg)
            
            # Store conversation history
            conversation_id = f"{workflow_type}_{datetime.now().timestamp()}"
            self.conversation_histories[conversation_id] = {
                "messages": [
                    {"role": "user", "content": task_msg},
                    {"role": "assistant", "content": response}
                ],
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
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "workflow_type": workflow_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup MCP server connections"""
        print("Cleaning up MCP servers...")
        for server in self._mcp_server_contexts:
            try:
                await server.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing server: {e}")
        print("✓ MCP servers closed")


# Singleton instance
_agents_manager_instance = None


async def get_agents_manager() -> AgentsManager:
    """Get or create the singleton AgentsManager instance"""
    global _agents_manager_instance
    if _agents_manager_instance is None:
        _agents_manager_instance = AgentsManager()
        await _agents_manager_instance.initialize()
    return _agents_manager_instance