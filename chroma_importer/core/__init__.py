"""
Core functionality for the Chroma Importer package.

This module provides the base classes and utilities for document processing,
embedding generation, and ChromaDB integration.
"""

from chroma_importer.core.base_processor import BaseDocumentProcessor
from chroma_importer.core.chroma_manager import ChromaManager

__all__ = ["BaseDocumentProcessor", "ChromaManager"]
