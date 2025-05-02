#!/usr/bin/env python3
"""
Command-line interface for the Chroma Importer package.

This module provides a unified CLI with subcommands for different document formats.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from chroma_importer.core import ChromaManager
# Import the processor factory functions, not the classes directly
from chroma_importer.processors import PDFProcessor, JavadocProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Set up common arguments for all processors.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    # Collection arguments
    parser.add_argument(
        "--collection", "-c",
        help="Name of the Chroma collection"
    )

    parser.add_argument(
        "--persist-dir", "-d",
        default="./chroma_db",
        help="Directory to persist the Chroma database (default: ./chroma_db)"
    )

    # Bedrock arguments
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

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in each batch (default: 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )


def parse_pdf_args(subparsers) -> None:
    """Set up PDF-specific arguments.
    
    Args:
        subparsers: Subparsers object to add parser to
    """
    pdf_parser = subparsers.add_parser(
        "pdf", 
        help="Import PDF files to Chroma"
    )
    
    # Source arguments
    pdf_parser.add_argument(
        "--pdf", "-p",
        required=True,
        help="Path to the PDF file to import"
    )
    
    # PDF-specific arguments
    pdf_parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=2000,
        help="Minimum size for text chunks (default: 2000)"
    )

    pdf_parser.add_argument(
        "--fallback-to-standard",
        action="store_true",
        help="Fallback to standard chunking if TOC extraction fails"
    )
    
    # Common arguments
    setup_common_args(pdf_parser)
    
    # Default collection name
    pdf_parser.set_defaults(collection="pdf_collection")


def parse_javadoc_args(subparsers) -> None:
    """Set up Javadoc-specific arguments.
    
    Args:
        subparsers: Subparsers object to add parser to
    """
    javadoc_parser = subparsers.add_parser(
        "javadoc", 
        help="Import Javadoc files to Chroma"
    )
    
    # Source arguments
    javadoc_parser.add_argument(
        "--javadoc", "-j",
        required=True,
        help="Path to the Javadoc directory to import"
    )
    
    # Common arguments
    setup_common_args(javadoc_parser)
    
    # Default collection name
    javadoc_parser.set_defaults(collection="javadoc_collection")


def process_pdf(args) -> int:
    """Process PDF files based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Check if PDF exists
        if not os.path.isfile(args.pdf):
            logger.error(f"PDF file not found: {args.pdf}")
            return 1

        # Create persist directory if it doesn't exist
        os.makedirs(args.persist_dir, exist_ok=True)

        logger.info(f"Processing PDF file: {args.pdf}")
        
        # Initialize and process PDF
        processor = PDFProcessor(
            args.pdf,
            min_chunk_size=args.min_chunk_size,
            fallback_to_standard=args.fallback_to_standard,
            verbose=args.verbose
        )
        
        # Process documents
        documents = processor.process_and_validate()
        
        if not documents:
            logger.error("No documents extracted from PDF")
            return 1
            
        # Initialize ChromaManager and add documents
        logger.info(f"Adding {len(documents)} documents to ChromaDB")
        
        chroma_manager = ChromaManager(
            args.persist_dir,
            args.model_id,
            args.region,
            args.profile
        )
        
        chroma_manager.add_documents_incrementally(
            documents,
            args.collection,
            args.batch_size
        )

        logger.info("PDF successfully imported to Chroma")
        return 0

    except Exception as e:
        logger.error(f"Error importing PDF: {str(e)}", exc_info=True)
        return 1


def process_javadoc(args) -> int:
    """Process Javadoc directory based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Check if Javadoc directory exists
        if not os.path.isdir(args.javadoc):
            logger.error(f"Javadoc directory not found: {args.javadoc}")
            return 1

        # Create persist directory if it doesn't exist
        os.makedirs(args.persist_dir, exist_ok=True)

        logger.info(f"Processing Javadoc directory: {args.javadoc}")
        
        # Initialize and process Javadoc
        processor = JavadocProcessor(
            args.javadoc,
            verbose=args.verbose
        )
        
        # Process documents
        documents = processor.process_and_validate()
        
        if not documents:
            logger.error("No documents extracted from Javadoc")
            return 1
            
        # Initialize ChromaManager and add documents
        logger.info(f"Adding {len(documents)} documents to ChromaDB")
        
        chroma_manager = ChromaManager(
            args.persist_dir,
            args.model_id,
            args.region,
            args.profile
        )
        
        chroma_manager.add_documents_incrementally(
            documents,
            args.collection,
            args.batch_size
        )

        logger.info("Javadoc successfully imported to Chroma")
        return 0

    except Exception as e:
        logger.error(f"Error importing Javadoc: {str(e)}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Import documents to ChromaDB using Amazon Bedrock embeddings"
    )
    
    # Create subparsers for different document types
    subparsers = parser.add_subparsers(
        dest="command",
        help="Document type to process",
        required=True
    )
    
    # Register document type subparsers
    parse_pdf_args(subparsers)
    parse_javadoc_args(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process based on command
    if args.command == "pdf":
        return process_pdf(args)
    elif args.command == "javadoc":
        return process_javadoc(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
