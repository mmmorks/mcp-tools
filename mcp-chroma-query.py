#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3>=1.37.24",
#     "chromadb>=0.4.18",
#     "fastmcp>=0.1.0",
#     "langchain>=0.0.267",
#     "langchain-aws>=0.2.22",
#     "langchain-community>=0.0.10",
#     "langchain-core>=0.3.56",
#     "pydantic>=2.4.2",
#     "pymupdf>=1.23.7",
#     "pypdf>=3.15.1",
#     "uvicorn>=0.24.0",
# ]
# ///

"""
FastMCP Chroma STDIN Server

This script provides a Model Context Protocol (MCP) server for Claude to query 
Chroma vector collections through STDIN/STDOUT using the FastMCP framework.

Usage:
    python mcp-chroma-query.py [--verbose] [--persist-dir DIR] [--model-id ID] [--region REGION]
"""

import sys
import argparse
import logging
from typing import Dict, Any, Optional, List

from fastmcp import FastMCP

# Import from the original query script
from query_chroma_collection import ChromaCollectionQuerier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Global configuration that will be set from command line arguments
config = {
    "persist_dir": "./chroma_db",
    "model_id": "amazon.titan-embed-text-v1",
    "region": "us-east-1",
    "aws_profile": None,
    "mmr_lambda": 0.5,
    "include_scores": True,
    "max_content_length": 0
}

# Global ChromaCollectionQuerier instance
querier = None

mcp = FastMCP(
    name="chroma-query-fastmcp",
    instructions="MCP Server for querying Chroma vector collections with Bedrock embeddings via FastMCP"
)

# Define tool functions with annotations
@mcp.tool()
def query_collection(
    query: str, 
    collection: str = "pdf_collection", 
    k: int = 5, 
    filter_dict: Optional[Dict[str, Any]] = None,
    search_type: str = "similarity"
) -> Dict[str, Any]:
    """Query a Chroma collection with semantic search using Bedrock embeddings
    
    Args:
        query: The query text to search for
        collection: Name of the Chroma collection
        k: Number of results to return
        filter_dict: Metadata filter dictionary (optional)
        search_type: Search type: similarity or mmr (Maximum Marginal Relevance)
    
    Returns:
        Search results in MCP context format
    """
    try:
        global querier
        
        # Check if the global querier is for a different collection
        if querier is None or querier.collection_name != collection:
            logger.info(f"Creating new ChromaCollectionQuerier for collection '{collection}'")
            querier = ChromaCollectionQuerier(
                collection_name=collection,
                persist_dir=config["persist_dir"],
                model_id=config["model_id"],
                region=config["region"],
                aws_profile=config["aws_profile"]
            )
        
        # Use the querier instance
        result = querier.query(
            query=query,
            k=k,
            filter_dict=filter_dict,
            search_type=search_type,
            mmr_lambda=config["mmr_lambda"],
            include_scores=config["include_scores"],
            max_content_length=config["max_content_length"]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

@mcp.tool()
def list_collections() -> Dict[str, List[Dict[str, Any]]]:
    """List available Chroma collections in the specified directory
    
    Returns:
        Dictionary containing a list of collections with their metadata
    """
    try:
        import chromadb
        
        # Initialize Chroma client using the configured persist_dir
        client = chromadb.PersistentClient(path=config["persist_dir"])
        collections = client.list_collections()
        
        result = []
        for collection in collections:
            count = collection.count()
            result.append({
                "name": collection.name,
                "count": count,
                "metadata": collection.metadata
            })
        
        return {"collections": result}
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise

# Resource implementation with annotation
@mcp.resource("collection/{collection_name}")
def collection_info(collection_name: str) -> Dict[str, Any]:
    """Information about a specific Chroma collection
    
    Args:
        collection_name: Name of the collection to get info about
    
    Returns:
        Dictionary containing information about the collection
    """
    try:
        import chromadb
        
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=config["persist_dir"])
        
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Collection not found: {str(e)}")
        
        # Get collection info
        count = collection.count()
        
        # Get some sample document IDs
        peek_results = collection.peek(limit=5)
        sample_ids = peek_results.get('ids', [])
        
        return {
            "name": collection_name,
            "count": count,
            "metadata": collection.metadata,
            "sample_ids": sample_ids
        }
        
    except Exception as e:
        logger.error(f"Error handling collection info request: {str(e)}")
        raise

def main():
    """Main function to parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Start the FastMCP Chroma STDIN server"
    )
    
    # Add command-line arguments for all configuration options
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory where the Chroma database is persisted (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--model-id",
        default="amazon.titan-embed-text-v1",
        choices=["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
        help="Bedrock embedding model ID (default: amazon.titan-embed-text-v1)"
    )
    
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region for Bedrock (default: us-west-2)"
    )
    
    parser.add_argument(
        "--profile",
        help="AWS profile name (optional)"
    )
    
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.5,
        help="MMR lambda parameter (0-1). Higher values prioritize relevance, lower values prioritize diversity (default: 0.5)"
    )
    
    parser.add_argument(
        "--with-scores",
        action="store_true",
        default=True,
        help="Include similarity scores in the output (default: True)"
    )
    
    parser.add_argument(
        "--max-content",
        type=int,
        default=0,
        help="Maximum content length to display (default: 0 for unlimited)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Update global configuration from command line arguments
    global config
    config.update({
        "persist_dir": args.persist_dir,
        "model_id": args.model_id,
        "region": args.region,
        "aws_profile": args.profile,
        "mmr_lambda": args.mmr_lambda,
        "include_scores": args.with_scores,
        "max_content_length": args.max_content
    })
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        mcp.set_verbose(True)
    
    logger.info("Starting FastMCP Chroma STDIN server with annotation-based tools and resources")
    
    # Start the server
    mcp.run()

if __name__ == "__main__":
    main()
