#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3>=1.37.24",
#     "chromadb>=0.4.18",
#     "langchain>=0.0.267",
#     "langchain-aws>=0.2.22",
#     "langchain-community>=0.0.10",
#     "langchain-core>=0.3.56",
#     "pydantic>=2.4.2",
#     "pymupdf>=1.23.7",
#     "pypdf>=3.15.1",
# ]
# ///
"""
Chroma Collection Query Module

This script provides a simple interface to query Chroma vector collections
that were created using the Titan embedding model from Amazon Bedrock.
It supports various query types and filtering options, and can output in
formats compatible with the model context protocol for LLMs.

The module can be used:
1. From the command line with arguments
2. By reading JSON input from stdin (--stdin flag)
3. As a Python module by importing the ChromaCollectionQuerier class
"""

import argparse
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple

import chromadb
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ChromaCollectionQuerier:
    """
    Class for querying Chroma collections with Bedrock embeddings.
    Maintains stateful components like embedding function and collection connection.
    """
    
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        persist_dir: str = "./chroma_db",
        model_id: str = "amazon.titan-embed-text-v1",
        region: str = "us-west-2",
        aws_profile: Optional[str] = None
    ):
        """Initialize the ChromaCollectionQuerier with embedding model and collection."""
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.model_id = model_id
        self.region = region
        self.aws_profile = aws_profile
        
        # Initialize embedding function
        self.embedding_function = self._setup_embeddings()
        
        # Load Chroma collection
        self.chroma_db = self._load_chroma_collection()
    
    def _setup_embeddings(self) -> BedrockEmbeddings:
        """Initialize Bedrock embeddings with the specified model."""
        logger.info(f"Initializing Bedrock embeddings with model {self.model_id} in region {self.region}")
        
        bedrock_kwargs = {
            "model_id": self.model_id,
            "region_name": self.region
        }
        
        if self.aws_profile:
            bedrock_kwargs["credentials_profile_name"] = self.aws_profile
            logger.info(f"Using AWS profile: {self.aws_profile}")
        
        return BedrockEmbeddings(**bedrock_kwargs)
    
    def _load_chroma_collection(self) -> Chroma:
        """Load an existing Chroma collection."""
        logger.info(f"Loading Chroma collection '{self.collection_name}' from {self.persist_dir}")
        
        try:
            # Initialize Chroma client
            client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Check if collection exists
            try:
                client.get_collection(name=self.collection_name)
            except Exception as e:
                logger.error(f"Collection '{self.collection_name}' not found: {str(e)}")
                available_collections = client.list_collections()
                if available_collections:
                    collection_names = [c.name for c in available_collections]
                    logger.info(f"Available collections: {', '.join(collection_names)}")
                else:
                    logger.info("No collections available in this directory")
                raise ValueError(f"Collection '{self.collection_name}' not found")
            
            # Load the collection with LangChain's Chroma wrapper
            chroma_db = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_function
            )
            
            logger.info(f"Successfully loaded collection '{self.collection_name}'")
            return chroma_db
        
        except Exception as e:
            logger.error(f"Error loading Chroma collection: {str(e)}")
            raise
    
    def _format_results(self, results: Union[List[Document], List[Tuple[Document, float]]], 
                       include_scores: bool = False, 
                       max_content_length: int = 0) -> List[Dict[str, Any]]:
        """Format search results for display."""
        formatted_results = []
        
        for item in results:
            if include_scores:
                doc, score = item
                result = {
                    "score": float(score),
                    "content": doc.page_content if max_content_length <= 0 else 
                              (doc.page_content[:max_content_length] + 
                               ("..." if len(doc.page_content) > max_content_length else "")),
                    "metadata": doc.metadata
                }
            else:
                doc = item
                result = {
                    "content": doc.page_content if max_content_length <= 0 else 
                              (doc.page_content[:max_content_length] + 
                               ("..." if len(doc.page_content) > max_content_length else "")),
                    "metadata": doc.metadata
                }
            
            formatted_results.append(result)
        
        return formatted_results
    
    def _format_mcp_context(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Format results in Model Context Protocol format."""
        context_items = []
        
        for i, result in enumerate(results):
            # Create a context item for each result
            context_item = {
                "id": f"result_{i+1}",
                "content": result["content"],
                "metadata": {
                    "source": result["metadata"].get("title", "Unknown"),
                    "page_range": result["metadata"].get("page_range", "Unknown"),
                    "section_path": result["metadata"].get("section_path", "Unknown"),
                }
            }
            
            # Add score if available
            if "score" in result:
                context_item["metadata"]["relevance_score"] = result["score"]
                
            context_items.append(context_item)
        
        # Create the MCP context object
        mcp_context = {
            "query": query,
            "context_items": context_items
        }
        
        return mcp_context
    
    def query(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        search_type: str = "similarity",
        mmr_lambda: float = 0.5,
        include_scores: bool = True,
        max_content_length: int = 0
    ) -> Dict[str, Any]:
        """
        Query the collection and return results in Model Context Protocol format.
        """
        try:
            # Prepare search kwargs
            search_kwargs = {"k": k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            # Execute search based on search type
            if search_type == "similarity":
                if include_scores:
                    results = self.chroma_db.similarity_search_with_score(query, **search_kwargs)
                else:
                    results = self.chroma_db.similarity_search(query, **search_kwargs)
            else:  # mmr
                fetch_k = min(k * 4, 20)
                results = self.chroma_db.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda,
                    **({} if not filter_dict else {"filter": filter_dict})
                )
            
            # Format results
            formatted_results = self._format_results(results, include_scores, max_content_length)
            
            # Return MCP format
            return self._format_mcp_context(formatted_results, query)
        
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            # Return error in a structured format
            return {
                "query": query,
                "error": str(e),
                "context_items": []
            }


def parse_filter(filter_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse filter string into a dictionary."""
    if not filter_str:
        return None
    
    try:
        return json.loads(filter_str)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid filter JSON: {str(e)}")
        raise ValueError(f"Filter must be valid JSON. Error: {str(e)}")


# Legacy function for backwards compatibility
def query_collection(
    query: str,
    collection: str = "pdf_collection",
    persist_dir: str = "./chroma_db",
    model_id: str = "amazon.titan-embed-text-v1",
    region: str = "us-east-1",
    aws_profile: Optional[str] = None,
    k: int = 5,
    filter_dict: Optional[Dict] = None,
    search_type: str = "similarity",
    mmr_lambda: float = 0.5,
    include_scores: bool = True,
    max_content_length: int = 0
) -> Dict[str, Any]:
    """
    Legacy function for backwards compatibility.
    Creates a new ChromaCollectionQuerier instance for each call.
    """
    querier = ChromaCollectionQuerier(
        collection_name=collection,
        persist_dir=persist_dir,
        model_id=model_id,
        region=region,
        aws_profile=aws_profile
    )
    
    return querier.query(
        query=query,
        k=k,
        filter_dict=filter_dict,
        search_type=search_type,
        mmr_lambda=mmr_lambda,
        include_scores=include_scores,
        max_content_length=max_content_length
    )


def main():
    """Main function to parse arguments and run the query."""
    parser = argparse.ArgumentParser(
        description="Query a Chroma collection using Amazon Bedrock embeddings"
    )
    
    parser.add_argument(
        "--query", "-q",
        help="The query text to search for (required unless using --stdin)"
    )
    
    parser.add_argument(
        "--collection", "-c",
        default="pdf_collection",
        help="Name of the Chroma collection (default: pdf_collection)"
    )
    
    parser.add_argument(
        "--persist-dir", "-d",
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
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )
    
    parser.add_argument(
        "--profile",
        help="AWS profile name (optional)"
    )
    
    parser.add_argument(
        "--k", "-n",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--filter",
        help="Metadata filter in JSON format, e.g. '{\"level\": 1}'"
    )
    
    parser.add_argument(
        "--search-type",
        choices=["similarity", "mmr"],
        default="similarity",
        help="Search type: similarity or mmr (Maximum Marginal Relevance) (default: similarity)"
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
        help="Include similarity scores in the output"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["pretty", "json", "mcp"],
        default="pretty",
        help="Output format: pretty, json, or mcp (Model Context Protocol) (default: pretty)"
    )
    
    parser.add_argument(
        "--max-content",
        type=int,
        default=300,
        help="Maximum content length to display (default: 300, use 0 for unlimited)"
    )
    
    parser.add_argument(
        "--output-file",
        help="Write output to file instead of stdout"
    )
    
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read query parameters from stdin as JSON"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Handle stdin input mode
        if args.query:
            # Parse filter if provided
            filter_dict = parse_filter(args.filter)
            
            # Create a ChromaCollectionQuerier instance
            querier = ChromaCollectionQuerier(
                collection_name=args.collection,
                persist_dir=args.persist_dir,
                model_id=args.model_id,
                region=args.region,
                aws_profile=args.profile
            )
            
            # Process query
            result = querier.query(
                query=args.query,
                k=args.k,
                filter_dict=filter_dict,
                search_type=args.search_type,
                mmr_lambda=args.mmr_lambda,
                include_scores=args.with_scores or args.output_format == "mcp",
                max_content_length=args.max_content
            )
            
            # Format output based on requested format
            if args.output_format == "json":
                output = json.dumps(result["context_items"], indent=2)
            elif args.output_format == "mcp":
                output = json.dumps(result, indent=2)
            else:  # pretty
                output_lines = []
                for item in result["context_items"]:
                    i = int(item["id"].split("_")[1]) - 1
                    score_str = f" (Score: {item['metadata']['relevance_score']})" if "relevance_score" in item["metadata"] else ""
                    output_lines.append(f"\n--- Result {i+1}{score_str} ---")
                    output_lines.append(f"Content: {item['content']}")
                    output_lines.append("Metadata:")
                    for key, value in item["metadata"].items():
                        if key != "relevance_score":  # Already displayed in the header
                            output_lines.append(f"  {key}: {value}")
                output = "\n".join(output_lines)
            
            # Output results
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output)
                logger.info(f"Results written to {args.output_file}")
            else:
                print(output)
            
            logger.info(f"Found {len(result['context_items'])} results")
            return 0
        
        else:
            logger.error("--query must be provided")
            parser.print_help()
            return 1
    
    except Exception as e:
        logger.error(f"Error querying collection: {str(e)}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    exit(main())
else:
    # When imported as a module, expose both the class and legacy function
    __all__ = ['ChromaCollectionQuerier', 'query_collection']
