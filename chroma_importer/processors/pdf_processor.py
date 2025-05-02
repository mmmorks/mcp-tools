"""
PDF Processor for extracting content from PDF files.

This module provides functionality for extracting content from PDF files 
with support for TOC-based semantic chunking.
"""

import logging
from typing import List, Dict

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from chroma_importer.core.base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(BaseDocumentProcessor):
    """Process PDF files with optional TOC-based semantic chunking."""
    
    def __init__(
        self, 
        pdf_path: str, 
        min_chunk_size: int = 2000,
        fallback_to_standard: bool = True,
        verbose: bool = False
    ):
        """Initialize the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
            min_chunk_size: Minimum size for text chunks
            fallback_to_standard: Whether to fallback to standard chunking if TOC extraction fails
            verbose: Enable verbose logging
        """
        super().__init__(pdf_path, verbose)
        
        # Validate file is a PDF
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File {pdf_path} is not a PDF")
            
        self.min_chunk_size = min_chunk_size
        self.fallback_to_standard = fallback_to_standard
    
    def process(self) -> List[Document]:
        """Process the PDF file using TOC-based semantic chunking if available.
        
        Returns:
            A list of Document objects with content and metadata
        """
        try:
            # Try TOC extraction first
            self.logger.info(f"Extracting TOC from {self.source_path}")
            toc = self.extract_toc()
            
            if toc and len(toc) > 0:
                self.logger.info(f"Found TOC with {len(toc)} entries, using semantic chunking")
                return self.create_semantic_chunks(toc)
            elif self.fallback_to_standard:
                self.logger.warning("No TOC found, falling back to standard chunking")
                return self.create_standard_chunks()
            else:
                self.logger.warning("No TOC found and fallback disabled, returning empty list")
                return []
                
        except Exception as e:
            if self.fallback_to_standard:
                self.logger.warning(f"TOC extraction failed: {str(e)}. Falling back to standard chunking")
                return self.create_standard_chunks()
            else:
                raise
    
    def extract_toc(self) -> List[Dict]:
        """Extract table of contents from PDF with page numbers.
        
        Returns:
            A list of TOC entries with level, title and page number
        """
        doc = fitz.open(self.source_path)
        toc = doc.get_toc()
        
        # Format: [[level, title, page, ...], ...]
        formatted_toc = []
        for entry in toc:
            level, title, page = entry[:3]
            formatted_toc.append({
                "level": level,
                "title": title,
                "page": page
            })
        
        self.logger.info(f"Extracted {len(formatted_toc)} TOC entries")
        return formatted_toc
    
    def create_semantic_chunks(self, toc: List[Dict]) -> List[Document]:
        """Create semantic chunks based on TOC structure.
        
        Args:
            toc: List of TOC entries
            
        Returns:
            A list of Document objects with content and metadata
        """
        self.logger.info("Creating semantic chunks based on TOC structure")
        
        # Load the PDF
        doc = fitz.open(self.source_path)
        total_pages = len(doc)
        self.logger.info(f"PDF has {total_pages} pages")
        
        # Sort TOC by page number
        toc_sorted = sorted(toc, key=lambda x: x["page"])
        
        # Create page ranges for each section
        sections = []
        for i in range(len(toc_sorted)):
            start_page = toc_sorted[i]["page"] - 1  # Convert from 1-indexed to 0-indexed
            
            # End page is either the start of the next section or the end of the document
            end_page = toc_sorted[i+1]["page"] - 1 if i < len(toc_sorted) - 1 else total_pages - 1
            
            sections.append({
                "title": toc_sorted[i]["title"],
                "level": toc_sorted[i]["level"],
                "start_page": start_page,
                "end_page": end_page
            })
        
        self.logger.info(f"Created {len(sections)} section definitions from TOC")
        
        # Create chunks based on sections
        chunks = []
        total_sections = len(sections)
        
        for idx, section in enumerate(sections):
            # Log progress
            if idx % 10 == 0 or idx == total_sections - 1:
                self.logger.info(f"Processing section {idx+1}/{total_sections} ({(idx+1)/total_sections*100:.1f}%)")
            
            # Extract text from the section's pages
            section_text = ""
            for page_num in range(section["start_page"], section["end_page"] + 1):
                page = doc[page_num]
                section_text += page.get_text()
            
            # Create section path for hierarchical context
            section_path = section["title"]
            
            # If section is too large, split it further
            if len(section_text) > self.min_chunk_size * 2:  # Reduced threshold for large sections
                self.logger.debug(f"Section '{section['title']}' is large ({len(section_text)} chars), splitting further")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.min_chunk_size,
                    chunk_overlap=min(300, self.min_chunk_size // 5)
                )
                sub_chunks = text_splitter.create_documents(
                    [section_text], 
                    metadatas=[{
                        "title": section["title"],
                        "level": section["level"],
                        "page_range": f"{section['start_page']+1}-{section['end_page']+1}",
                        "hierarchy_path": section_path,
                        "element_type": "section"
                    }]
                )
                chunks.extend(sub_chunks)
            else:
                # Keep small sections as single chunks
                chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        "title": section["title"],
                        "level": section["level"],
                        "page_range": f"{section['start_page']+1}-{section['end_page']+1}",
                        "hierarchy_path": section_path,
                        "element_type": "section"
                    }
                ))
        
        self.logger.info(f"Created {len(chunks)} semantic chunks based on TOC structure")
        return chunks
    
    def create_standard_chunks(self) -> List[Document]:
        """Create standard chunks using PyPDFLoader and RecursiveCharacterTextSplitter.
        
        Returns:
            A list of Document objects with content and metadata
        """
        self.logger.info("Loading PDF with PyPDFLoader for standard chunking")
        
        # Load the PDF
        documents = PyPDFLoader(self.source_path).load()
        self.logger.info(f"Loaded {len(documents)} pages")
        
        # Apply standard chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.min_chunk_size,
            chunk_overlap=min(300, self.min_chunk_size // 5)
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add element_type to metadata
        for chunk in chunks:
            chunk.metadata["element_type"] = "page_chunk"
            
            # Create a hierarchy path from page numbers
            if "page" in chunk.metadata:
                chunk.metadata["hierarchy_path"] = f"page_{chunk.metadata['page']}"
        
        self.logger.info(f"Created {len(chunks)} chunks with standard chunking")
        return chunks
