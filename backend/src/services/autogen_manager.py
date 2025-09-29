# ============================================================================
# AUTOGEN CONFIGURATION
# ============================================================================
import json
from datetime import datetime
from typing import Dict, Any
from config.env_config import config as env
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

class AutoGenManager:
    def __init__(self):

        # Initialize OpenAI client for AutoGen
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.0-flash",
            api_key=env.GEMINI_API_KEY,
        )
        
        self.setup_agents()
    
    def setup_agents(self):
        """Setup specialized agents for different workflows"""
        
        # Coordinator Agent - Main orchestrator
        self.coordinator = AssistantAgent(
            name="coordinator",
            description="Main coordinator for workplace AI assistant that routes requests to specialists",
            model_client=self.model_client,
            system_message="""You are the main coordinator for a workplace AI assistant.
            Your role is to:
            1. Analyze incoming requests and route them to appropriate specialists
            2. Ensure proper workflow execution
            3. Coordinate between different agents
            4. Provide final responses to users
            
            Available workflows: standup, meeting, qa, onboarding, transcription
            Be concise and actionable in your responses.""",
        )
        
        # Standup Agent - Handles daily standups
        self.standup_agent = AssistantAgent(
            name="standup_specialist",
            description="Specialist for handling daily standups and team progress tracking",
            model_client=self.model_client,
            system_message="""You are a standup specialist. You help teams with:
            1. Collecting daily standup updates from team members
            2. Summarizing team progress and identifying patterns
            3. Identifying blockers and potential risks
            4. Creating actionable items from standup discussions
            
            Be concise, supportive, and focus on actionable insights.
            Format your responses in a clear, structured way.""",
        )
        
        # Meeting Agent - Handles meeting processing
        self.meeting_agent = AssistantAgent(
            name="meeting_specialist", 
            description="Specialist for processing meeting transcripts and extracting insights",
            model_client=self.model_client,
            system_message="""You are a meeting specialist. You help with:
            1. Processing meeting transcripts and recordings
            2. Extracting key decisions, action items, and outcomes
            3. Creating concise meeting summaries
            4. Identifying follow-up tasks and ownership
            
            Focus on clarity, accuracy, and actionable outcomes.
            Always identify specific action items with clear ownership.""",
        )
        
        # QA Agent - Handles knowledge queries
        self.qa_agent = AssistantAgent(
            name="qa_specialist",
            description="Knowledge specialist for answering questions using company knowledge base",
            model_client=self.model_client,
            system_message="""You are a knowledge specialist. You help with:
            1. Answering questions using available knowledge base and context
            2. Providing accurate, contextual information to users
            3. Suggesting related resources and documentation
            4. Learning from interactions to improve responses
            
            Be accurate, helpful, and cite sources when possible.
            If you don't know something, say so clearly and suggest alternatives.""",
        )
        
        # Onboarding Agent - Handles new hire onboarding  
        self.onboarding_agent = AssistantAgent(
            name="onboarding_specialist",
            description="Specialist for managing new hire onboarding processes and tasks",
            model_client=self.model_client,
            system_message="""You are an onboarding specialist. You help with:
            1. Creating personalized onboarding plans for new hires
            2. Tracking onboarding progress and completion status
            3. Providing guidance and support to new team members
            4. Ensuring smooth integration into the team and company culture
            
            Be welcoming, organized, and thorough in your approach.
            Break down complex processes into manageable steps.""",
        )
        
        # User Proxy for human interaction
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            description="Proxy agent for handling user interactions and inputs",
        )
    
    async def process_workflow(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow using appropriate agents"""
        try:
            # Select appropriate agents based on workflow type
            agents_map = {
                "standup": [self.coordinator, self.standup_agent],
                "meeting": [self.coordinator, self.meeting_agent], 
                "qa": [self.coordinator, self.qa_agent],
                "onboarding": [self.coordinator, self.onboarding_agent],
                "transcription": [self.coordinator, self.meeting_agent],
            }
            
            selected_agents = agents_map.get(workflow_type, [self.coordinator])
            
            # Create initial message
            initial_message = TextMessage(
                content=f"Process {workflow_type} workflow with the following data:\n{json.dumps(data, indent=2)}",
                source="user"
            )
            
            # Process with multiple agents if needed
            if len(selected_agents) > 1:

                # Create a round-robin group chat for multi-agent collaboration
                team = RoundRobinGroupChat(selected_agents)
                
                # Run the team collaboration
                result = await team.run(
                    task=initial_message.content,
                    termination_condition=lambda messages: len(messages) >= 5  # Limit rounds
                )
                
                # Extract the final response
                final_response = result.messages[-1].content if result.messages else "No response generated"
                
            else:
                # Single agent processing
                agent = selected_agents[0]
                response = await agent.on_messages([initial_message], cancellation_token=None)
                final_response = response.chat_message.content if response.chat_message else "No response generated"
            
            return {
                "status": "success",
                "workflow_type": workflow_type,
                "result": final_response,
                "agent_used": [agent.name for agent in selected_agents],
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
        """Process a simple query with the coordinator agent"""
        try:
            context_str = f"\nContext: {json.dumps(context)}" if context else ""
            message = TextMessage(
                content=f"{query}{context_str}",
                source="user"
            )
            
            response = await self.coordinator.on_messages([message], cancellation_token=None)
            return response.chat_message.content if response.chat_message else "No response generated"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

autogen_manager = AutoGenManager()
