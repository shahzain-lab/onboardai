"""
Pinecone MCP Server - A Model Context Protocol server for interacting with Pinecone vector database.

This server provides tools for querying knowledge bases and answering questions using Pinecone
with automatic text-to-embedding conversion via Pinecone Inference API.

Installation:
    pip install "mcp[cli]" pinecone

Usage:
    # Run with stdio transport
    python pinecone_mcp_server.py
    
    # Or with MCP CLI
    uv run mcp dev pinecone_mcp_server.py

Environment Variables Required:
    PINECONE_API_KEY: Your Pinecone API key
    PINECONE_INDEX_NAME: Name of your Pinecone index
    PINECONE_EMBEDDING_MODEL: Embedding model to use (optional, default: multilingual-e5-large)
"""

import os
from typing import Any, Optional
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP
from pinecone import Pinecone, ServerlessSpec


# Configuration dataclass
@dataclass
class PineconeConfig:
    """Configuration for Pinecone connection."""
    api_key: str
    index_name: str
    embedding_model: str = "multilingual-e5-large"
    cloud: str = "aws"
    region: str = "us-east-1"


# Application context with Pinecone client
@dataclass
class AppContext:
    """Application context with Pinecone dependencies."""
    pc: Pinecone
    index: Any
    config: PineconeConfig


# Lifespan management for Pinecone connection
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage Pinecone connection lifecycle."""
    # Load configuration from environment variables
    config = PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("PINECONE_INDEX_NAME", "onboardai-kb"),
        embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1")
    )
    
    if not config.api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    if not config.index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable is required")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=config.api_key)
    
    # Get or create index with inference integration
    index_name = config.index_name
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_exists = any(idx['name'] == index_name for idx in existing_indexes)
    
    if not index_exists:
        # Create index with integrated embedding model
        pc.create_index(
            name=index_name,
            dimension=1024,  # dimension for multilingual-e5-large
            metric='cosine',
            spec=ServerlessSpec(
                cloud=config.cloud,
                region=config.region
            )
        )
    
    # Connect to index
    index = pc.Index(index_name)
    
    try:
        yield AppContext(pc=pc, index=index, config=config)
    finally:
        # Cleanup if needed
        pass


# Create FastMCP server with lifespan
mcp = FastMCP(
    name="Pinecone Knowledge Base Server",
    lifespan=app_lifespan
)


@mcp.tool()
async def kb_query(
    query_text: str,
    top_k: int = 5,
    namespace: str = "",
    include_metadata: bool = True,
    filter_dict: Optional[dict] = None,
    input_type: str = "query",
    ctx: Context = None
) -> dict[str, Any]:
    """
    Query the Pinecone knowledge base with text (automatic embedding conversion).
    
    Args:
        query_text: The query text to search for
        top_k: Number of top results to return (default: 5)
        namespace: Pinecone namespace to query (default: "" for default namespace)
        include_metadata: Whether to include metadata in results (default: True)
        filter_dict: Optional metadata filter dictionary
        input_type: Type of input - "query" or "passage" (default: "query")
        ctx: MCP context (automatically injected)
    
    Returns:
        Dictionary containing:
        - matches: List of matching vectors with scores and metadata
        - namespace: The namespace queried
        - query_text: Original query text
    
    Example:
        # Query with text
        results = kb_query(
            query_text="What is machine learning?",
            top_k=3,
            namespace="documents",
            filter_dict={"category": "technology"}
        )
    """
    try:
        # Get Pinecone client and index from context
        app_ctx = ctx.request_context.lifespan_context
        pc = app_ctx.pc
        index = app_ctx.index
        embedding_model = app_ctx.config.embedding_model
        
        await ctx.info(f"Querying with text: '{query_text[:100]}...'")
        
        # Generate embedding using Pinecone Inference API
        embedding_response = pc.inference.embed(
            model=embedding_model,
            inputs=[query_text],
            parameters={
                "input_type": input_type,
                "truncate": "END"
            }
        )
        
        # Extract the embedding vector
        query_vector = embedding_response[0].values
        
        await ctx.debug(f"Generated embedding with dimension: {len(query_vector)}")
        
        # Perform query
        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=False,
            filter=filter_dict
        )
        
        # Convert response to dict
        result = {
            "query_text": query_text,
            "matches": [
                {
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": match.metadata if include_metadata else None
                }
                for match in query_response.matches
            ],
            "namespace": namespace
        }
        
        await ctx.info(f"Query returned {len(result['matches'])} results")
        return result
        
    except Exception as e:
        await ctx.error(f"Error querying Pinecone: {str(e)}")
        return {
            "error": str(e),
            "query_text": query_text,
            "matches": [],
            "namespace": namespace
        }


@mcp.tool()
async def kb_answer_qa(
    question: str,
    top_k: int = 3,
    namespace: str = "",
    context_key: str = "text",
    filter_dict: Optional[dict] = None,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Answer a question using the knowledge base context retrieved from Pinecone.
    
    This tool automatically converts the question to embeddings using Pinecone Inference API,
    queries Pinecone for relevant context, and returns the top matching documents.
    
    Args:
        question: The question to answer
        top_k: Number of context documents to retrieve (default: 3)
        namespace: Pinecone namespace to search (default: "" for default namespace)
        context_key: Metadata key containing the text content (default: "text")
        filter_dict: Optional metadata filter dictionary
        ctx: MCP context (automatically injected)
    
    Returns:
        Dictionary containing:
        - question: Original question
        - answer_context: List of relevant text contexts from the knowledge base
        - sources: List of source documents with metadata
        - scores: Relevance scores for each context
        - total_results: Number of results retrieved
    
    Example:
        # Answer a question using the knowledge base
        result = kb_answer_qa(
            question="What is machine learning?",
            top_k=3,
            namespace="wiki",
            context_key="content"
        )
    """
    try:
        # Get Pinecone client and index from context
        app_ctx = ctx.request_context.lifespan_context
        pc = app_ctx.pc
        index = app_ctx.index
        embedding_model = app_ctx.config.embedding_model
        
        await ctx.info(f"Answering question: '{question[:100]}...'")
        
        # Generate embedding for the question using Pinecone Inference API
        embedding_response = pc.inference.embed(
            model=embedding_model,
            inputs=[question],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )
        
        # Extract the embedding vector
        query_vector = embedding_response[0].values
        
        # Query Pinecone for relevant context
        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            include_values=False,
            filter=filter_dict
        )
        
        # Extract context and sources
        answer_contexts = []
        sources = []
        scores = []
        
        for match in query_response.matches:
            # Extract text content from metadata
            if match.metadata and context_key in match.metadata:
                context_text = match.metadata[context_key]
                answer_contexts.append(context_text)
                scores.append(float(match.score))
                
                # Store source information
                source_info = {
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": {k: v for k, v in match.metadata.items() if k != context_key}
                }
                sources.append(source_info)
        
        result = {
            "question": question,
            "answer_context": answer_contexts,
            "sources": sources,
            "scores": scores,
            "total_results": len(answer_contexts),
            "namespace": namespace
        }
        
        await ctx.info(f"Retrieved {len(answer_contexts)} relevant contexts for answering")
        
        if len(answer_contexts) == 0:
            await ctx.warning("No relevant context found in knowledge base")
            result["message"] = "No relevant context found to answer the question"
        
        return result
        
    except Exception as e:
        await ctx.error(f"Error in kb_answer_qa: {str(e)}")
        return {
            "question": question,
            "error": str(e),
            "answer_context": [],
            "sources": [],
            "scores": [],
            "total_results": 0,
            "namespace": namespace
        }


@mcp.tool()
async def kb_upsert_text(
    documents: list[dict],
    namespace: str = "",
    batch_size: int = 100,
    input_type: str = "passage",
    ctx: Context = None
) -> dict[str, Any]:
    """
    Upsert text documents into Pinecone with automatic embedding generation.
    
    This tool automatically converts text to embeddings using Pinecone Inference API
    before upserting to the vector database.
    
    Args:
        documents: List of document dictionaries, each containing:
                   - id: Unique identifier (string)
                   - text: Text content to embed (string)
                   - metadata: Optional metadata dictionary
        namespace: Pinecone namespace (default: "" for default namespace)
        batch_size: Number of documents to process per batch (default: 100)
        input_type: Type of input - "passage" or "query" (default: "passage")
        ctx: MCP context (automatically injected)
    
    Returns:
        Dictionary containing:
        - upserted_count: Number of documents successfully upserted
        - namespace: The namespace used
        - status: Operation status
    
    Example:
        # Upsert text documents
        result = kb_upsert_text(
            documents=[
                {
                    "id": "doc1",
                    "text": "Machine learning is a subset of AI...",
                    "metadata": {"title": "ML Intro", "category": "education"}
                },
                {
                    "id": "doc2", 
                    "text": "Deep learning uses neural networks...",
                    "metadata": {"title": "DL Basics", "category": "education"}
                }
            ],
            namespace="knowledge_base"
        )
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        pc = app_ctx.pc
        index = app_ctx.index
        embedding_model = app_ctx.config.embedding_model
        
        await ctx.info(f"Upserting {len(documents)} text documents to namespace '{namespace}'")
        
        total_upserted = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract texts for embedding
            texts = [doc["text"] for doc in batch]
            
            # Generate embeddings using Pinecone Inference API
            await ctx.debug(f"Generating embeddings for batch of {len(texts)} documents")
            embedding_response = pc.inference.embed(
                model=embedding_model,
                inputs=texts,
                parameters={
                    "input_type": input_type,
                    "truncate": "END"
                }
            )
            
            # Prepare vectors for upsert
            vectors = []
            for doc, embedding in zip(batch, embedding_response):
                vector_data = {
                    "id": doc["id"],
                    "values": embedding.values,
                }
                
                # Add metadata if provided
                if "metadata" in doc and doc["metadata"]:
                    # Store the text in metadata for later retrieval
                    metadata = doc["metadata"].copy()
                    metadata["text"] = doc["text"]
                    vector_data["metadata"] = metadata
                else:
                    vector_data["metadata"] = {"text": doc["text"]}
                
                vectors.append(vector_data)
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)
            total_upserted += len(vectors)
            
            # Report progress
            progress = total_upserted / len(documents)
            await ctx.report_progress(
                progress=progress,
                total=1.0,
                message=f"Upserted {total_upserted}/{len(documents)} documents"
            )
        
        result = {
            "upserted_count": total_upserted,
            "namespace": namespace,
            "status": "success",
            "embedding_model": embedding_model
        }
        
        await ctx.info(f"Successfully upserted {total_upserted} documents using {embedding_model}")
        return result
        
    except Exception as e:
        await ctx.error(f"Error upserting documents: {str(e)}")
        return {
            "upserted_count": 0,
            "namespace": namespace,
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
async def kb_stats(
    namespace: str = "",
    ctx: Context = None
) -> dict[str, Any]:
    """
    Get statistics about the Pinecone index.
    
    Args:
        namespace: Optional namespace to get stats for specific namespace
        ctx: MCP context (automatically injected)
    
    Returns:
        Dictionary containing index statistics including:
        - dimension: Vector dimension
        - index_fullness: How full the index is
        - total_vector_count: Total number of vectors
        - namespaces: Statistics per namespace
        - embedding_model: The embedding model being used
    
    Example:
        stats = kb_stats(namespace="documents")
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        index = app_ctx.index
        embedding_model = app_ctx.config.embedding_model
        
        await ctx.info("Fetching index statistics")
        
        stats = index.describe_index_stats()
        
        result = {
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "total_vector_count": stats.total_vector_count,
            "embedding_model": embedding_model,
            "namespaces": {}
        }
        
        # Add namespace-specific stats
        if hasattr(stats, 'namespaces') and stats.namespaces:
            for ns_name, ns_stats in stats.namespaces.items():
                result["namespaces"][ns_name] = {
                    "vector_count": ns_stats.vector_count
                }
        
        await ctx.info("Successfully retrieved index statistics")
        return result
        
    except Exception as e:
        await ctx.error(f"Error getting index stats: {str(e)}")
        return {
            "error": str(e),
            "dimension": 0,
            "total_vector_count": 0
        }


@mcp.tool()
async def kb_delete(
    ids: Optional[list[str]] = None,
    delete_all: bool = False,
    namespace: str = "",
    filter_dict: Optional[dict] = None,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Delete vectors from the Pinecone index.
    
    Args:
        ids: List of vector IDs to delete (optional)
        delete_all: If True, delete all vectors in the namespace (default: False)
        namespace: Pinecone namespace (default: "" for default namespace)
        filter_dict: Optional metadata filter for conditional deletion
        ctx: MCP context (automatically injected)
    
    Returns:
        Dictionary containing:
        - status: Operation status
        - namespace: The namespace used
        - deleted_ids: List of deleted IDs (if applicable)
    
    Example:
        # Delete specific documents
        result = kb_delete(
            ids=["doc1", "doc2"],
            namespace="documents"
        )
        
        # Delete all documents in a namespace
        result = kb_delete(
            delete_all=True,
            namespace="documents"
        )
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        index = app_ctx.index
        
        if delete_all:
            await ctx.warning(f"Deleting ALL vectors in namespace '{namespace}'")
            index.delete(delete_all=True, namespace=namespace)
            result = {
                "status": "success",
                "namespace": namespace,
                "operation": "delete_all"
            }
            await ctx.info(f"Deleted all vectors in namespace '{namespace}'")
            
        elif ids:
            await ctx.info(f"Deleting {len(ids)} vectors from namespace '{namespace}'")
            index.delete(ids=ids, namespace=namespace)
            result = {
                "status": "success",
                "namespace": namespace,
                "deleted_ids": ids,
                "count": len(ids)
            }
            await ctx.info(f"Deleted {len(ids)} vectors")
            
        elif filter_dict:
            await ctx.info(f"Deleting vectors matching filter in namespace '{namespace}'")
            index.delete(filter=filter_dict, namespace=namespace)
            result = {
                "status": "success",
                "namespace": namespace,
                "operation": "delete_by_filter",
                "filter": filter_dict
            }
            await ctx.info("Deleted vectors matching filter")
            
        else:
            result = {
                "status": "error",
                "error": "Must provide either 'ids', 'delete_all=True', or 'filter_dict'"
            }
        
        return result
        
    except Exception as e:
        await ctx.error(f"Error deleting vectors: {str(e)}")
        return {
            "status": "error",
            "namespace": namespace,
            "error": str(e)
        }


# Main entry point
if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")