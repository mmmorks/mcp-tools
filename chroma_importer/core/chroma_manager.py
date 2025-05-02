"""
ChromaManager for handling interactions with ChromaDB.

This module provides functionality for embedding generation and storage in ChromaDB.
"""

import time
import logging
from typing import List, Optional
from datetime import datetime

import chromadb
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages interactions with ChromaDB and embedding generation."""
    
    def __init__(
        self,
        persist_dir: str,
        model_id: str,
        region_name: str,
        aws_profile: Optional[str] = None,
    ):
        """Initialize the ChromaManager.
        
        Args:
            persist_dir: Directory to persist the Chroma database
            model_id: Bedrock embedding model ID
            region_name: AWS region for Bedrock
            aws_profile: AWS profile name (optional)
        """
        self.persist_dir = persist_dir
        self.model_id = model_id
        self.region_name = region_name
        self.aws_profile = aws_profile
        
        # Initialize embedding model
        logger.info(f"Initializing Bedrock embeddings with model {model_id} in region {region_name}")
        bedrock_kwargs = {
            "model_id": model_id,
            "region_name": region_name
        }

        if aws_profile:
            bedrock_kwargs["credentials_profile_name"] = aws_profile
            logger.info(f"Using AWS profile: {aws_profile}")

        self.embeddings = BedrockEmbeddings(**bedrock_kwargs)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_dir)
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            A ChromaDB collection object
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
            
        return collection
    
    def add_documents_incrementally(
        self,
        documents: List[Document],
        collection_name: str,
        batch_size: int = 10,
    ) -> chromadb.Collection:
        """Add documents to ChromaDB incrementally.
        
        Args:
            documents: List of Document objects to add
            collection_name: Name of the collection to add to
            batch_size: Number of documents to process in each batch
            
        Returns:
            The ChromaDB collection with added documents
        """
        start_time = time.time()
        collection = self.get_or_create_collection(collection_name)
        
        # Process chunks in batches
        total_chunks = len(documents)
        total_batches = (total_chunks - 1) // batch_size + 1
        successful_chunks = 0
        failed_chunks = 0
        
        logger.info(f"Processing {total_chunks} documents in {total_batches} batches (batch size: {batch_size})")
        
        for i in range(0, total_chunks, batch_size):
            batch = documents[i:i+batch_size]
            end_idx = min(i+batch_size, total_chunks)
            batch_num = i//batch_size + 1
            
            batch_start_time = time.time()
            logger.info(f"Processing batch {batch_num}/{total_batches} (chunks {i+1}-{end_idx} of {total_chunks})")
            
            # Create IDs for the documents
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{j}" for j in range(i, end_idx)]
            
            # Extract text and metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Generate embeddings and add to collection
            try:
                logger.info(f"Generating embeddings for batch {batch_num}...")
                embedding_start = time.time()
                embeddings = self.embeddings.embed_documents(texts)
                embedding_time = time.time() - embedding_start
                logger.info(f"Embeddings generated in {embedding_time:.2f} seconds")
                
                logger.info(f"Adding documents to Chroma collection...")
                add_start = time.time()
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                add_time = time.time() - add_start
                logger.info(f"Documents added in {add_time:.2f} seconds")
                
                successful_chunks += len(batch)
                
                batch_time = time.time() - batch_start_time
                logger.info(f"Batch {batch_num}/{total_batches} completed in {batch_time:.2f} seconds")
                
                # Estimate remaining time
                avg_time_per_batch = (time.time() - start_time) / batch_num
                remaining_batches = total_batches - batch_num
                est_remaining_time = avg_time_per_batch * remaining_batches
                
                # Format as hours:minutes:seconds
                est_completion_time = datetime.now().timestamp() + est_remaining_time
                est_completion_str = datetime.fromtimestamp(est_completion_time).strftime('%H:%M:%S')
                
                logger.info(f"Progress: {batch_num}/{total_batches} batches ({batch_num/total_batches*100:.1f}%)")
                logger.info(f"Estimated time remaining: {est_remaining_time/60:.1f} minutes (completion around {est_completion_str})")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                failed_chunks += len(batch)
                # Continue with next batch instead of failing completely
                continue
        
        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Successfully added {successful_chunks} chunks to Chroma collection '{collection_name}'")
        
        if failed_chunks > 0:
            logger.warning(f"Failed to process {failed_chunks} chunks")
        
        return collection
