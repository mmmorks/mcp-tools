"""
Javadoc Processor for extracting content from Javadoc HTML files.

This module provides functionality for traversing a Javadoc directory structure,
extracting content from HTML files, and preserving semantic relationships.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from chroma_importer.core.base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class JavadocProcessor(BaseDocumentProcessor):
    """Process Javadoc HTML files with semantic structure awareness."""
    
    def __init__(
        self,
        javadoc_path: str,
        verbose: bool = False
    ):
        """Initialize the Javadoc processor.
        
        Args:
            javadoc_path: Path to the Javadoc directory
            verbose: Enable verbose logging
        """
        super().__init__(javadoc_path, verbose)
        
        # Validate path is a directory
        if not os.path.isdir(javadoc_path):
            raise ValueError(f"Javadoc path {javadoc_path} is not a directory")
            
        # Check if this looks like a Javadoc structure
        index_html = os.path.join(javadoc_path, "index.html")
        if not os.path.exists(index_html):
            raise ValueError(f"No index.html found in {javadoc_path}, doesn't appear to be a Javadoc directory")
        
        self.javadoc_path = Path(javadoc_path)
    
    def process(self) -> List[Document]:
        """Process the Javadoc directory structure.
        
        Returns:
            A list of Document objects with content and metadata
        """
        self.logger.info(f"Starting Javadoc processing from {self.javadoc_path}")
        
        # Extract package structure
        package_structure = self.extract_package_structure()
        all_documents = []
        
        self.logger.info(f"Processing {len(package_structure)} packages")
        
        # Process each package
        for package_name, files in package_structure.items():
            self.logger.info(f"Processing package {package_name} with {len(files)} files")
            
            # Process package summary
            package_summary_path = self.javadoc_path / "/".join(package_name.split(".")) / "package-summary.html"
            if package_summary_path.exists():
                package_doc = self.process_package_summary(package_summary_path, package_name)
                if package_doc:
                    all_documents.append(package_doc)
            
            # Process each file in the package
            with ThreadPoolExecutor(max_workers=min(10, len(files))) as executor:
                future_to_file = {
                    executor.submit(self.process_file, file_path, package_name): file_path 
                    for file_path in files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        documents = future.result()
                        all_documents.extend(documents)
                    except Exception as e:
                        self.logger.error(f"Exception processing {file_path}: {e}")
        
        self.logger.info(f"Extracted {len(all_documents)} documents from Javadoc")
        return all_documents
    
    def get_all_html_files(self) -> List[Path]:
        """Get all HTML files in the Javadoc directory.
        
        Returns:
            A list of Path objects for relevant HTML files
        """
        html_files = []
        for path in self.javadoc_path.glob("**/*.html"):
            if not path.is_file():
                continue
                
            # Skip index-all.html, deprecated-list.html, etc.
            if path.name.startswith(("index-", "deprecated", "help-doc", 
                                      "overview-tree", "overview-summary",
                                      "allclasses", "allpackages", "constant-values")):
                continue
                
            html_files.append(path)
        
        self.logger.info(f"Found {len(html_files)} HTML files for processing")
        return html_files
    
    def extract_package_structure(self) -> Dict[str, List[Path]]:
        """Extract package structure from Javadoc.
        
        Returns:
            Dictionary mapping package names to lists of file paths
        """
        package_files = {}
        all_html_files = self.get_all_html_files()
        
        # Find package-summary.html files to identify packages
        package_summaries = [f for f in all_html_files if f.name == "package-summary.html"]
        
        for package_file in package_summaries:
            # The parent directory name is the package name
            package_path = package_file.parent
            package_name = self._extract_package_name(package_file)
            
            # Find all HTML files in this package
            package_files[package_name] = [
                f for f in all_html_files 
                if f.parent == package_path and f.name != "package-summary.html"
            ]
        
        self.logger.info(f"Extracted structure for {len(package_files)} packages")
        return package_files
    
    def _extract_package_name(self, package_file: Path) -> str:
        """Extract package name from package-summary.html.
        
        Args:
            package_file: Path to package-summary.html
            
        Returns:
            Package name string
        """
        try:
            with open(package_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html5lib')
                
                # Try to find package name from header
                header = soup.select_one("div.header h1.title")
                if header:
                    # Extract package name from "Package packagename"
                    package_text = header.text.strip()
                    if package_text.startswith("Package "):
                        return package_text[8:].strip()
                
                # Fallback: extract from the file path
                rel_path = package_file.parent.relative_to(self.javadoc_path)
                package_parts = []
                
                # If there's a multi-level directory, convert to package name
                for part in rel_path.parts:
                    # Skip common doc structure directories
                    if part in ["api", "javadoc", "docs"]:
                        continue
                    package_parts.append(part)
                
                if package_parts:
                    return ".".join(package_parts)
                
                # If we can't determine a package name
                return "(default package)"
                
        except Exception as e:
            self.logger.warning(f"Error extracting package name from {package_file}: {e}")
            return f"unknown_package_{package_file.parent.name}"
    
    def process_file(self, file_path: Path, package_name: str) -> List[Document]:
        """Process a single Javadoc HTML file and extract documents.
        
        Args:
            file_path: Path to the HTML file
            package_name: Name of the package containing the file
            
        Returns:
            List of Document objects extracted from the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html5lib')
            documents = []
            
            # Determine file type (class, interface, enum, etc.)
            file_type = self._determine_file_type(soup)
            class_name = file_path.stem
            
            # Get class summary
            class_doc = self._extract_class_documentation(soup, class_name, package_name, file_type)
            if class_doc:
                documents.append(class_doc)
            
            # Get method documentation
            method_docs = self._extract_methods_documentation(soup, class_name, package_name, file_type)
            documents.extend(method_docs)
            
            # Get field documentation
            field_docs = self._extract_fields_documentation(soup, class_name, package_name, file_type)
            documents.extend(field_docs)
            
            self.logger.debug(f"Extracted {len(documents)} document chunks from {file_path}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_package_summary(self, package_file: Path, package_name: str) -> Optional[Document]:
        """Extract package summary documentation.
        
        Args:
            package_file: Path to package-summary.html
            package_name: Name of the package
            
        Returns:
            Document object with package summary or None
        """
        try:
            with open(package_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            soup = BeautifulSoup(content, 'html5lib')
            
            # Find package description
            package_desc = soup.select_one("div.block")
            if not package_desc:
                return None
                
            description = package_desc.text.strip()
            if not description:
                return None
                
            content = f"PACKAGE: {package_name}\n\n{description}"
            
            return Document(
                page_content=content,
                metadata={
                    "package": package_name,
                    "element_type": "package_summary",
                    "element_name": package_name,
                    "hierarchy_path": package_name,
                    "full_name": package_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing package summary {package_file}: {e}")
            return None
    
    def _determine_file_type(self, soup: BeautifulSoup) -> str:
        """Determine the file type (class, interface, enum).
        
        Args:
            soup: BeautifulSoup object for the HTML file
            
        Returns:
            File type string (class, interface, enum, annotation)
        """
        # Check the title or header for the file type
        header = soup.select_one("div.header h1.title, div.title")
        if header:
            text = header.text.strip().lower()
            if "interface" in text:
                return "interface"
            elif "enum" in text:
                return "enum"
            elif "annotation" in text:
                return "annotation"
            elif "class" in text:
                return "class"
            
        # Look for specific indicators
        if soup.select_one(".interfaceName"):
            return "interface"
        elif soup.select_one(".enumDescription"):
            return "enum"
        elif soup.select_one(".annotationDescription"):
            return "annotation"
            
        # Default to class
        return "class"
    
    def _extract_class_documentation(self, soup: BeautifulSoup, class_name: str, 
                                   package_name: str, file_type: str) -> Optional[Document]:
        """Extract class/interface summary documentation.
        
        Args:
            soup: BeautifulSoup object for the HTML file
            class_name: Name of the class
            package_name: Name of the package
            file_type: Type of the file (class, interface, etc.)
            
        Returns:
            Document object with class documentation or None
        """
        # Get the class description
        class_description = soup.select_one("div.description div.block, div.contentContainer div.description div.block")
        if not class_description:
            class_description = soup.select_one("div.classSummary div.block")
            
        # Get inheritance information
        inheritance_section = soup.select_one("div.inheritance, ul.inheritance")
        inheritance = ""
        if inheritance_section:
            inheritance = " ".join(inheritance_section.text.strip().split())
            
        # Check if deprecated
        deprecated_section = soup.select_one("div.deprecatedContent")
        deprecated_note = ""
        if deprecated_section:
            deprecated_note = "DEPRECATED: " + " ".join(deprecated_section.text.strip().split())
            
        # Combine information
        content = f"{file_type.upper()}: {class_name}\n\n"
        
        if class_description:
            content += class_description.text.strip() + "\n\n"
            
        if inheritance:
            content += f"Inheritance: {inheritance}\n\n"
            
        if deprecated_note:
            content += f"{deprecated_note}\n\n"
            
        # Only create a document if we have actual content
        if len(content.strip()) > len(f"{file_type.upper()}: {class_name}"):
            return Document(
                page_content=content,
                metadata={
                    "package": package_name,
                    "class_name": class_name,
                    "type": file_type,
                    "element_type": "class_summary",
                    "element_name": class_name,
                    "hierarchy_path": f"{package_name}.{class_name}",
                    "full_name": f"{package_name}.{class_name}"
                }
            )
        return None
    
    def _extract_methods_documentation(self, soup: BeautifulSoup, class_name: str, 
                                     package_name: str, file_type: str) -> List[Document]:
        """Extract documentation for methods.
        
        Args:
            soup: BeautifulSoup object for the HTML file
            class_name: Name of the class
            package_name: Name of the package
            file_type: Type of the file (class, interface, etc.)
            
        Returns:
            List of Document objects with method documentation
        """
        documents = []
        
        # Find all method detail sections
        method_sections = soup.select("section.detail")
        
        for section in method_sections:
            # Check if this is a method section (not field or constructor)
            if not section.select_one("h3:-soup-contains('Method Detail')"):
                continue
                
            # Find all method details
            methods = section.select("ul.blockList > li.blockList")
            
            for method in methods:
                method_name = None
                method_signature = None
                method_desc = None
                
                # Get method name and signature
                h4 = method.select_one("h4")
                if h4:
                    method_name = h4.text.strip()
                    
                pre = method.select_one("pre")
                if pre:
                    method_signature = pre.text.strip()
                    
                # Get method description
                desc = method.select_one("div.block")
                if desc:
                    method_desc = desc.text.strip()
                    
                if method_name and (method_signature or method_desc):
                    # Clean up method name (remove any type parameters)
                    method_name = re.sub(r"<.*>", "", method_name).strip()
                    
                    content = f"METHOD: {method_name}\n\n"
                    
                    if method_signature:
                        content += f"Signature: {method_signature}\n\n"
                        
                    if method_desc:
                        content += f"Description: {method_desc}\n\n"
                        
                    # Extract parameter docs
                    param_tags = method.select("dt:-soup-contains('Parameters:') ~ dd")
                    if param_tags:
                        content += "Parameters:\n"
                        for param in param_tags:
                            param_text = param.text.strip()
                            param_name = param.select_one("span.paramName")
                            if param_name:
                                param_name = param_name.text.strip()
                                param_desc = param_text.replace(param_name, "", 1).strip()
                                content += f"- {param_name}: {param_desc}\n"
                            else:
                                content += f"- {param_text}\n"
                        content += "\n"
                        
                    # Extract return doc
                    return_tags = method.select("dt:-soup-contains('Returns:') ~ dd")
                    if return_tags and return_tags[0]:
                        content += f"Returns: {return_tags[0].text.strip()}\n\n"
                        
                    # Extract throws docs
                    throws_tags = method.select("dt:-soup-contains('Throws:') ~ dd")
                    if throws_tags:
                        content += "Throws:\n"
                        for throws in throws_tags:
                            content += f"- {throws.text.strip()}\n"
                        content += "\n"
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "package": package_name,
                            "class_name": class_name,
                            "type": file_type,
                            "element_type": "method",
                            "element_name": method_name,
                            "hierarchy_path": f"{package_name}.{class_name}.{method_name}",
                            "full_name": f"{package_name}.{class_name}.{method_name}"
                        }
                    ))
        
        return documents
    
    def _extract_fields_documentation(self, soup: BeautifulSoup, class_name: str, 
                                    package_name: str, file_type: str) -> List[Document]:
        """Extract documentation for fields.
        
        Args:
            soup: BeautifulSoup object for the HTML file
            class_name: Name of the class
            package_name: Name of the package
            file_type: Type of the file (class, interface, etc.)
            
        Returns:
            List of Document objects with field documentation
        """
        documents = []
        
        # Find field detail section
        field_section = soup.select_one("section.field-details, section.detail:has(h3:-soup-contains('Field Detail'))")
        if not field_section:
            return []
            
        fields = field_section.select("ul.blockList > li.blockList")
        
        for field in fields:
            field_name = None
            field_type = None
            field_desc = None
            
            # Get field name
            h4 = field.select_one("h4")
            if h4:
                field_name = h4.text.strip()
                
            # Get field type and declaration
            pre = field.select_one("pre")
            if pre:
                field_declaration = pre.text.strip()
                # Try to extract type from declaration
                match = re.search(r'(\w+(?:<.*>)?)\s+\w+', field_declaration)
                if match:
                    field_type = match.group(1)
                    
            # Get field description
            desc = field.select_one("div.block")
            if desc:
                field_desc = desc.text.strip()
                
            if field_name:
                content = f"FIELD: {field_name}\n\n"
                
                if field_type:
                    content += f"Type: {field_type}\n\n"
                    
                if field_desc:
                    content += f"Description: {field_desc}\n\n"
                    
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "package": package_name,
                        "class_name": class_name,
                        "type": file_type,
                        "element_type": "field",
                        "element_name": field_name,
                        "hierarchy_path": f"{package_name}.{class_name}.{field_name}",
                        "full_name": f"{package_name}.{class_name}.{field_name}"
                    }
                ))
                
        return documents
