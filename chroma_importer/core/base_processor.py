"""
Base Processor for document processing.

This module provides an abstract base class for document processors that defines 
the common interface and functionality.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, source_path: str, verbose: bool = False):
        """Initialize the document processor.
        
        Args:
            source_path: Path to the document source (file or directory)
            verbose: Enable verbose logging
        """
        self.source_path = source_path
        self.verbose = verbose
        
        # Validate source path exists
        if not os.path.exists(source_path):
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Set up logging level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        self.logger.info(f"Initialized {self.__class__.__name__} for {source_path}")
    
    @property
    def source_type(self) -> str:
        """Get the source type identifier.
        
        Returns:
            A string identifier for the processor type (e.g., 'pdf', 'javadoc')
        """
        return self.__class__.__name__.replace('Processor', '').lower()
    
    @abstractmethod
    def process(self) -> List[Document]:
        """Process the document source and extract content.
        
        Returns:
            A list of Document objects with content and metadata
        """
        pass
    
    def enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """Enrich document metadata with standard fields.
        
        Args:
            documents: List of Document objects to enrich
            
        Returns:
            The same list with enriched metadata
        """
        for doc in documents:
            # Add common metadata fields
            doc.metadata["source_path"] = self.source_path
            doc.metadata["source_type"] = self.source_type
            doc.metadata["processed_at"] = time.time()
            
            # Convert any non-string metadata values to strings for Chroma
            for key, value in doc.metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    doc.metadata[key] = str(value)
        
        return documents
    
    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate processed documents.
        
        Args:
            documents: List of Document objects to validate
            
        Returns:
            List of valid Document objects
        """
        valid_docs = []
        
        for i, doc in enumerate(documents):
            # Check for empty content
            if not doc.page_content or len(doc.page_content.strip()) == 0:
                self.logger.warning(f"Skipping document {i} with empty content")
                continue
                
            # Ensure metadata exists
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                self.logger.warning(f"Adding empty metadata to document {i}")
                doc.metadata = {}
                
            valid_docs.append(doc)
            
        self.logger.info(f"Validated {len(valid_docs)} documents out of {len(documents)}")
        return valid_docs
    
    def process_and_validate(self) -> List[Document]:
        """Process documents and validate them.
        
        Returns:
            List of valid Document objects with enriched metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing documents from {self.source_path}")
        
        try:
            # Process documents
            documents = self.process()
            
            # Validate and enrich
            valid_docs = self.validate_documents(documents)
            enriched_docs = self.enrich_metadata(valid_docs)
            
            processing_time = time.time() - start_time
            self.logger.info(
                f"Processed {len(enriched_docs)} documents in {processing_time:.2f} seconds"
            )
            
            return enriched_docs
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            raise
