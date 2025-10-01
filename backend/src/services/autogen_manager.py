# ============================================================================
# AGENTS MANAGER - OpenAI Agents SDK with Gemini + MCP
# ============================================================================
import json
from datetime import datetime
from typing import Dict, Any, List

from config.env_config import config as env
from services.openai_agents import MCPToolManager, GeminiClient

# OpenAI Agents SDK
from agents import Agent, Runner 


class AgentsManager:
    """
    Manages OpenAI Agents SDK agents with Gemini backend and MCP tools.
    """
    
    def __init__(self):
        self.gemini_api_key = env.GEMINI_API_KEY
        self.model = "gemini-2.0-flash-exp"
        
        # Initialize components
        self.gemini_client = GeminiClient(self.gemini_api_key, self.model)
        self.mcp_manager = MCPToolManager()
        self.runner = Runner()
        
        # Agents
        self.coordinator = None
        self.standup_agent = None
        self.meeting_agent = None
        self.qa_agent = None
        self.onboarding_agent = None
        self.summarizer_agent = None
        
        # Store conversation histories
        self.conversation_histories = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize MCP servers and create agents"""
        if self._initialized:
            return
        
        print("Initializing MCP servers...")
        await self.mcp_manager.initialize_mcp_servers()
        
        print("Creating OpenAI Agents with Gemini backend + MCP tools...")
        await self._setup_agents()
        
        self._initialized = True
        print("OpenAI Agents SDK + Gemini + MCP system ready!")
    
    async def _setup_agents(self):
        """Setup agents using OpenAI Agents SDK with Gemini and MCP tools"""
        
        # Get tools from MCP for each agent
        all_tools = self.mcp_manager.get_tools_for_agent()
        slack_db_tools = self.mcp_manager.get_tools_for_agent(["slack", "database"])
        google_slack_db_tools = self.mcp_manager.get_tools_for_agent([ "database"])
        print("All MCP Tools => => ", all_tools)

        # Create handoffs between agents
        handoff_to_standup = Agent(
            name="standup_specialist",
            handoff_description="Transfer to standup specialist for processing daily updates"
        )
        handoff_to_meeting = Agent(
            name="meeting_specialist",
            handoff_description="Transfer to meeting specialist for processing meeting transcripts"
        )
        handoff_to_qa = Agent(
            name="qa_specialist",
            handoff_description="Transfer to QA specialist for knowledge base queries",
        )
        handoff_to_onboarding = Agent(
            name="onboarding_specialist",
            handoff_description="Transfer to onboarding specialist for new hire workflows"
        )
        
        # 1. Coordinator Agent - Main orchestrator
        self.coordinator = Agent(
            name="coordinator",
            model=self.model,
            instructions="""You are the main coordinator for a workplace AI assistant.
            
            You have access to ALL tools via MCP:
            - Slack: Send messages, read channels, manage users
            - Google Calendar: Schedule meetings, check availability
            - Database: Query and store information
            - Filesystem: Read documentation
            
            Your responsibilities:
            1. Analyze incoming requests and determine the best approach
            2. Use tools directly for simple tasks
            3. Hand off complex tasks to specialist agents using the handoff functions
            4. Coordinate between multiple agents when needed
            
            Available specialists (use handoffs):
            - standup_specialist: For daily standup processing
            - meeting_specialist: For meeting transcripts and summaries
            - qa_specialist: For knowledge base queries
            - onboarding_specialist: For new hire onboarding
            
            Be concise and actionable. Complete tasks efficiently.""",
            mcp_servers=all_tools,
            handoffs=[handoff_to_standup, handoff_to_meeting, handoff_to_qa, handoff_to_onboarding]
        )
        
        # 2. Standup Agent - Specialized for standups
        self.standup_agent = Agent(
            name="standup_specialist",
            model=self.model,
            instructions="""You are a standup specialist for team updates.
            
            You have access to:
            - Slack: Send standup summaries and notifications
            - Database: Store and retrieve standup data
            
            Your workflow:
            1. Process standup data (yesterday, today, blockers)
            2. Store in database using available MCP tools
            3. Generate a clear, actionable summary
            4. Send notifications to relevant team members via Slack
            5. Identify any critical blockers that need attention

            Be concise, supportive, and focus on actionable insights.
            Format responses in a clear, structured way.""",
            mcp_servers=slack_db_tools,
        )
        
        # 3. Meeting Agent - Specialized for meetings
        self.meeting_agent = Agent(
            name="meeting_specialist",
            model=self.model,
            instructions="""You are a meeting specialist.
            
            You have access to:
            - Google Calendar: Access meeting details and participants
            - Slack: Send meeting summaries and notifications
            - Database: Store meeting data and action items
         
            Your workflow:
            1. Process meeting transcript
            2. Extract key decisions, action items, and important discussions
            3. Store everything in database with proper categorization
            4. Send concise summary to all participants via Slack
            5. Update calendar events if needed
            
            Focus on clarity, accuracy, and actionable outcomes.
            Always identify specific action items with clear ownership.""",
            mcp_servers=google_slack_db_tools,
        )
        
        # 4. QA Agent - Knowledge base specialist
        self.qa_agent = Agent(
            name="qa_specialist",
            model=self.model,
            instructions="""You are a knowledge specialist.
            
            You have access to:
            - Database: Search knowledge base and retrieve documents
            - Database: Read tasks and track progress
            - Filesystem: Read documentation files and resources
            
            Your workflow:
            1. Understand the question thoroughly
            2. Search database for relevant information
            3. Read relevant documentation files if needed
            4. Provide accurate, well-cited answers
            5. Suggest related resources and follow-up topics
            
            Be accurate, helpful, and always cite your sources.
            If you don't know something, say so clearly and suggest alternatives.""",
            mcp_servers=all_tools,
        )
        
        # 5. Onboarding Agent - New hire specialist
        self.onboarding_agent = Agent(
            name="onboarding_specialist",
            model=self.model,
            instructions="""You are an onboarding specialist for new hires.
            
            You have access to:
            - Database: Read onboarding tasks and track progress
            - Filesystem: Access onboarding documents and materials
            
            Your workflow:
            1. Create personalized onboarding plan based on role
            2. Send warm welcome message via Slack
            3. Store all tasks in database with due dates
            4. Provide onboarding resources and documentation
            5. Track progress and send reminders
            
            Be welcoming, organized, and thorough.
            Break down complex processes into manageable steps.""",
            mcp_servers=all_tools,
        )
        
        # 5. Onboarding Agent - New hire specialist
        self.summarizer_agent = Agent(
            name="summarizer_specialist",
            model=self.model,
            instructions="""You are a summarization specialist. Your ONLY job is to create ultra-concise, 
                single-line summaries for end users. 

                Rules:
                1. Generate ONLY ONE LINE (maximum 120 characters).
                2. Be extremely concise and plain, like a quick status update.
                3. Focus on the user’s task or outcome (what was done, what’s next).
                4. Do NOT mention system names, agents, MCP, routing, or backend steps.
                5. Avoid all technical terms (API, workflow, coordinator, etc.).
                6. Remove reasoning, explanations, and process details.
                7. No multiple sentences, no bullet points, no filler.
                8. Always sound like a short update in a daily standup context.""",
            mcp_servers=all_tools,
        )
        
    
    async def _run_agent_with_gemini(self, agent: Agent, messages: List[Dict]) -> str:
        """Run an agent with Gemini as the LLM backend"""
        try:
            response = await self.gemini_client.generate_response(messages, agent.mcp_servers)
            return response
        except Exception as e:
            return f"Error running agent: {str(e)}"
    
    async def process_workflow(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow using OpenAI Agents SDK with Gemini backend.
        Agents can hand off to each other as needed.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Select starting agent based on workflow type
            agent_map = {
                "standup": self.standup_agent,
                "meeting": self.meeting_agent,
                "qa": self.qa_agent,
                "onboarding": self.onboarding_agent,
                "transcription": self.meeting_agent,
            }
            
            # Start with coordinator for complex workflows, specialist for simple ones
            if workflow_type in ["standup", "meeting", "onboarding", "transcription"]:
                starting_agent = self.coordinator
            else:
                starting_agent = agent_map.get(workflow_type, self.qa_agent)

            print("starting_agent => => ", starting_agent)
            
            # Create task description
            task_description = f"""Process {workflow_type} workflow.

Data:
{json.dumps(data, indent=2)}

Please process this using your available MCP tools. Use handoffs to specialist agents if needed."""
            
            # Build conversation history
            conversation_id = f"{workflow_type}_{datetime.now().timestamp()}"
            messages = [
                {"role": "user", "content": task_description}
            ]
            
            # Run agent with Gemini
            response = await self._run_agent_with_gemini(starting_agent, messages)
            
            # Generate concise summary using summarizer agent
            summary_messages = [
                {"role": "user", "content": f"Summarize this in ONE LINE (max 150 chars):\n\n{response}"}
            ]
            summary = await self._run_agent_with_gemini(self.summarizer_agent, summary_messages)
            
            # Store conversation
            self.conversation_histories[conversation_id] = {
                "messages": messages + [{"role": "assistant", "content": response}],
                "agent": starting_agent.name,
                "workflow_type": workflow_type
            }
            
            return {
                "status": "success",
                "workflow_type": workflow_type,
                "result": response,  # Full detailed response
                "summary": summary.strip(),  # Concise 1-line summary
                "agent_used": starting_agent.name,
                "conversation_id": conversation_id,
                "architecture": "OpenAI Agents SDK → Gemini → MCP",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "workflow_type": workflow_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_simple_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process a simple query with the coordinator"""
        if not self._initialized:
            await self.initialize()
        
        try:
            context_str = f"\nContext: {json.dumps(context)}" if context else ""
            messages = [
                {"role": "user", "content": f"{query}{context_str}"}
            ]
            
            response = await self._run_agent_with_gemini(self.coordinator, messages)
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    async def continue_conversation(self, conversation_id: str, message: str) -> str:
        """Continue an existing conversation"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if conversation_id not in self.conversation_histories:
                return "Conversation not found"
            
            history = self.conversation_histories[conversation_id]
            agent_name = history["agent"]
            
            # Get the agent
            agent_attr_map = {
                "coordinator": self.coordinator,
                "standup_specialist": self.standup_agent,
                "meeting_specialist": self.meeting_agent,
                "qa_specialist": self.qa_agent,
                "onboarding_specialist": self.onboarding_agent,
            }
            agent = agent_attr_map.get(agent_name, self.coordinator)
            
            # Add user message
            history["messages"].append({"role": "user", "content": message})
            
            # Generate response
            response = await self._run_agent_with_gemini(agent, history["messages"])
            
            # Add assistant response
            history["messages"].append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"Error continuing conversation: {str(e)}"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_agents_manager_instance = None

async def get_agents_manager() -> AgentsManager:
    """Get or create agents manager singleton"""
    global _agents_manager_instance
    if _agents_manager_instance is None:
        _agents_manager_instance = AgentsManager()
        await _agents_manager_instance.initialize()
    return _agents_manager_instance