---
name: RAG Chroma
description: Retrieval-Augmented Generation (RAG) assistant using ChromaDB and a language model (LLM) to query curated Computational Materials Science resources.
---
# RAG Chroma Guidelines

## Overview

RAG Chroma is a Retrieval-Augmented Generation (RAG) toolset that indexes, searches, and chats with computational materials science curated resources using **ChromaDB** and a **Large Language Model (LLM)** (via LangChain and LangGraph). It supports local dense vector search and cloud-based hybrid (dense + sparse keyword RRF) search, as well as a conversational agent loop that dynamically calls search tools.

## Capabilities

* **Markdown Ingestion:** Parses markdown tables from lists of materials science resources and uploads them to a local or cloud-hosted Chroma collection with vector schemas.
* **Hybrid Search:** Queries the Chroma DB collection with support for dense vector distances, sparse keyword BM25 indexes, filtering by sections, and tag filtering.
* **Conversational Agent:** Executes an interactive command-line agent using a tool-calling capable LLM that leverages tool calling to answer technical queries using the ingested materials science database.

## How to use

Ensure the LLM backend (supporting tool calling) is running, then run the tasks from `skills/rag-chroma` using Pixi:

1. **Ingest Database:**
   ```bash
   # Ingest the default repository README.md
   pixi run ingest
   
   # Or specify a custom file
   pixi run ingest --file ../../README.md
   ```
2. **Execute Search:**
   ```bash
   # Search using defaults
   pixi run search
   
   # Search with arguments
   pixi run search --query "DFT software" --limit 5 --sections "Tools: Crystal structures"
   ```
3. **Interactive RAG Chat:**
   ```bash
   pixi run chat
   ```
4. **Run Verification Tests:**
   ```bash
   pixi run test
   ```

## Scripts

* [chroma_utils.py](file:///mnt/d/project/RSE/rag/cms-collection/skills/rag-chroma/scripts/chroma_utils.py): Centralized module containing connections, collections creation, markdown parser, and query search execution.
* [ingest_md.py](file:///mnt/d/project/RSE/rag/cms-collection/skills/rag-chroma/scripts/ingest_md.py): Ingestion CLI task.
* [chroma_search.py](file:///mnt/d/project/RSE/rag/cms-collection/skills/rag-chroma/scripts/chroma_search.py): Search CLI task.
* [ollama_chat.py](file:///mnt/d/project/RSE/rag/cms-collection/skills/rag-chroma/scripts/ollama_chat.py): Chat agent loop task.

## Best practices

1. **Lazy Initialization:** Ensure database clients, collections, and LLM instances are initialized lazily inside functions (rather than at the module import level) to keep imports fast and testable.
2. **Robust Argument Parsing:** Use recursive unwrappers (`unwrap_to_string`) inside tool calls to handle cases where local LLMs pass structured dictionaries or nested JSON objects instead of plain strings.
3. **Relative Path Resolution:** Resolve local directories (like `chroma_db`) relative to the package root directory rather than script directories to ensure consistency across execution environments.
