# ============================================================================
# MCP TOOL MANAGER AND GEMINI CLIENT
# ============================================================================
import json
import os
import asyncio
from typing import Dict, Any, Optional, List, Callable

# MCP (Model Context Protocol)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Google Generative AI for Gemini
from google import genai 
from google.genai import types
from config.env_config import config as env

class MCPToolManager:
    """Manages MCP servers and converts them to callable Python functions for OpenAI Agents SDK"""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, Any] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.connection_errors: Dict[str, str] = {}
        self._session_contexts = {}  # Store context managers to keep sessions alive
    
    async def initialize_mcp_servers(self):
        """Initialize MCP servers and keep connections alive"""
        db_server_path = os.path.join(
            os.path.dirname(__file__), "..", "servers", "database_tools.py"
        )
        kb_server_path = os.path.join(
            os.path.dirname(__file__), "..", "servers", "kb_vector_tools.py"
        )

        mcp_servers = { 
            "database": StdioServerParameters(
                command="python",
                args=[db_server_path],
                env={"DATABASE_URL": env.DATABASE_URL}
            ),
            "knowledge_base": StdioServerParameters(
                command="python",
                args=[kb_server_path],
                env={
                    "PINECONE_API_KEY": env.PINECONE_API_KEY,
                 }
            )
        }
            
        for name, params in mcp_servers.items():
            try:
                # Create persistent connection context
                stdio_ctx = stdio_client(params)
                read, write = await stdio_ctx.__aenter__()
                
                # Create persistent session
                session_ctx = ClientSession(read, write)
                session = await session_ctx.__aenter__()
                
                # Initialize the session
                await session.initialize()
                
                # Store contexts to keep them alive
                self._session_contexts[name] = {
                    'stdio': stdio_ctx,
                    'session_ctx': session_ctx,
                    'read': read,
                    'write': write
                }
                self.sessions[name] = session
                
                # Get available tools
                tools_response = await session.list_tools()
                
                for tool in tools_response.tools:
                    tool_key = f"{name}_{tool.name}"
                    
                    # Save full tool metadata
                    self.available_tools[tool_key] = {
                        "server": name,
                        "tool": tool,  
                        "description": tool.description,
                    }
                    
                    # Create executable wrapper with proper OpenAI function format
                    fn = self._create_tool_function(name, tool.name, tool.description, tool.inputSchema)
                    
                    # Store function with metadata in OpenAI format
                    self.tool_functions[tool_key] = fn
                
                print(f"✓ Connected to MCP server: {name} ({len(tools_response.tools)} tools)")
                
            except Exception as e:
                error_msg = f"Failed to connect to MCP server {name}: {str(e)}"
                print(f"✗ {error_msg}")
                self.connection_errors[name] = str(e)
    
    def _create_tool_function(self, server_name: str, tool_name: str, description: str, input_schema: dict) -> Callable:
        """Create a Python function that executes an MCP tool in OpenAI Agents SDK format"""
        
        async def tool_function(**kwargs) -> Dict[str, Any]:
            """Dynamically generated MCP tool function"""
            try:
                if server_name not in self.sessions:
                    return {
                        "error": f"MCP server '{server_name}' not connected",
                        "details": self.connection_errors.get(server_name, "Unknown error")
                    }
                
                session = self.sessions[server_name]
                
                # Call the MCP tool
                result = await session.call_tool(tool_name, kwargs)
                
                # Extract content from MCP result
                if hasattr(result, 'content') and result.content:
                    # Parse text content from MCP response
                    content_text = ""
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text
                        elif hasattr(content_item, 'type') and content_item.type == 'text':
                            content_text += str(content_item)
                    
                    # Try to parse as JSON if possible
                    try:
                        parsed_data = json.loads(content_text)
                        return parsed_data
                    except json.JSONDecodeError:
                        # Return as is if not JSON
                        return {"result": content_text}
                
                # Fallback: return the raw result
                return {"result": str(result)}
                
            except Exception as e:
                print(f"Error executing tool {tool_name}: {str(e)}")
                return {
                    "error": str(e), 
                    "tool": tool_name,
                    "server": server_name
                }
        
        # Set function metadata for OpenAI Agents SDK
        tool_function.__name__ = f"{server_name}_{tool_name}".replace("-", "_")
        tool_function.__doc__ = description or f"Execute {tool_name} on {server_name}"
        
        # Add schema information as function attributes (OpenAI SDK may use this)
        tool_function.input_schema = input_schema
        tool_function.description = description
        
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
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        for name, contexts in self._session_contexts.items():
            try:
                if 'session_ctx' in contexts:
                    await contexts['session_ctx'].__aexit__(None, None, None)
                if 'stdio' in contexts:
                    await contexts['stdio'].__aexit__(None, None, None)
                print(f"✓ Closed MCP server: {name}")
            except Exception as e:
                print(f"✗ Error closing {name}: {e}")

class GeminiClient:
    """Gemini client for generating responses and handling tools"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
    
    def _convert_messages(self, messages: List[Dict]) -> List[types.Content]:
        """Converts dictionary messages to Gemini's Content format."""
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg.get("content", "")
            
            if content:
                 # Alternatively, use types.Part.from_text(text=content)
                text_part = types.Part.from_text(text=content) 

                gemini_messages.append(
                    types.Content(role=role, parts=[text_part])
                )
        return gemini_messages

    async def generate_response(self, messages: List[Dict], tools: List[Any] = None) -> str:
        """
        Generate a response using Gemini, correctly passing tool declarations.
        """
        try:
            gemini_messages = self._convert_messages(messages)
            
            if not gemini_messages:
                return "Error generating response: No message content provided."

            # The last message is the current prompt, the rest is history
            # The current message (last element) is also passed as part of contents
            history = gemini_messages
            
            # Use generate_content for a single, stateless turn with history and tools.
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=history,  # Full conversation history
                config=types.GenerateContentConfig(
                    tools=tools if tools else None # Correctly passes tools
                )
            )
            
            # Check for tool calls and return a structured response if necessary
            if response.function_calls:
                return f"Function Call: {response.function_calls}"
            
            return response.text
            
        except Exception as e:
            # Note: For production, you'd want to handle specific API errors
            return f"Error generating response: {type(e).__name__}: {str(e)}"