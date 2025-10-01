# ============================================================================
# MCP TOOL MANAGER AND GEMINI CLIENT
# ============================================================================
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable

# MCP (Model Context Protocol)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Google Generative AI for Gemini
import google.generativeai as genai

from config.env_config import config as env


class MCPToolManager:
    """Manages MCP servers and converts them to callable Python functions"""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, Any] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.connection_errors: Dict[str, str] = {}
    
    async def initialize_mcp_servers(self):
        """Initialize MCP servers (without Slack)"""
        mcp_servers = {
            # "google": StdioServerParameters(
            #     command="npx",
            #     args=["-y", "@modelcontextprotocol/server-google-calendar"],
            #     env={
            #         "GOOGLE_CLIENT_ID": env.GOOGLE_CLIENT_ID,
            #         "GOOGLE_CLIENT_SECRET": env.GOOGLE_CLIENT_SECRET,
            #     }
            # ),
            "database": StdioServerParameters(
                command="server-postgres", 
                env={"DATABASE_URL": env.DATABASE_URL}
            ),
            "filesystem": StdioServerParameters(
                command="server-filesystem"
            ),
        }
        
        for name, params in mcp_servers.items():
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.sessions[name] = session
                        
                        tools_response = await session.list_tools()
                        for tool in tools_response.tools:
                            tool_key = f"{name}_{tool.name}"
                            self.available_tools[tool_key] = {
                                "server": name,
                                "tool": tool,
                                "description": tool.description,
                            }
                            
                            self.tool_functions[tool_key] = self._create_tool_function(
                                name, tool.name, tool.description
                            )
                        
                        print(f"✓ Connected to MCP server: {name} ({len(tools_response.tools)} tools)")
            except Exception as e:
                error_msg = f"Failed to connect to MCP server {name}: {str(e)}"
                print(f"✗ {error_msg}")
                self.connection_errors[name] = str(e)
    
    def _create_tool_function(self, server_name: str, tool_name: str, description: str) -> Callable:
        """Create a Python function that executes an MCP tool"""
        async def tool_function(**kwargs) -> str:
            """Dynamically generated MCP tool function"""
            try:
                if server_name not in self.sessions:
                    return json.dumps({
                        "error": f"MCP server '{server_name}' not connected",
                        "details": self.connection_errors.get(server_name, "Unknown error")
                    })
                
                session = self.sessions[server_name]
                result = await session.call_tool(tool_name, kwargs)
                return json.dumps({"success": True, "data": result})
            except Exception as e:
                return json.dumps({"error": str(e), "tool": tool_name})
        
        tool_function.__name__ = f"{server_name}_{tool_name}".replace("-", "_")
        tool_function.__doc__ = description or f"Execute {tool_name} on {server_name}"
        
        return tool_function
    
    def get_tools_for_agent(self, server_names: Optional[List[str]] = None) -> List[Callable]:
        """Get tool functions for OpenAI Agents SDK"""
        tools = []
        
        for tool_key, tool_func in self.tool_functions.items():
            server_name = tool_key.split("_")[0]
            
            if server_names and server_name not in server_names:
                continue
            
            tools.append(tool_func)
        
        return tools
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all MCP connections"""
        return {
            "connected": list(self.sessions.keys()),
            "failed": self.connection_errors,
            "total_tools": len(self.available_tools)
        }


class GeminiClient:
    """Gemini client for generating responses"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
    
    async def generate_response(self, messages: List[Dict], tools: List = None) -> str:
        """Generate response using Gemini"""
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get("content", "")
                gemini_messages.append({"role": role, "parts": [content]})
            
            # Start or continue chat
            chat = self.model.start_chat(
                history=gemini_messages[:-1] if len(gemini_messages) > 1 else []
            )
            
            # Send message and get response
            last_content = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            response = await asyncio.to_thread(chat.send_message, last_content)
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"