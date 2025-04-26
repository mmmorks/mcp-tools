# MCP Tools

A collection of tools for MCP (Model Context Protocol) operations.

## Overview

This repository contains utilities and tools for working with Model Context Protocol, focusing on document processing and vector database integration.

## Features

### import-pdf-to-chroma-v2
This tool allows you to import PDF documents into a Chroma vector database. It processes PDF content and stores it as embeddings in Chroma, using Amazon Bedrock for generating the embeddings.

### mcp-chroma-query.py
This program enables querying the Chroma vector database through an MCP (Model Context Protocol) integration. It connects your vector database to the LLM of your choice, allowing for semantic search and retrieval of document content.

These tools create a complete workflow for document processing, embedding generation, and retrieval augmented generation (RAG) using the Chroma vector database (https://docs.trychroma.com/docs/overview/getting-started) and Amazon Bedrock.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/mmmorks/mcp-tools.git
cd mcp-tools

# Install uv (dependency manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Chroma using uv
uv pip install chromadb
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
