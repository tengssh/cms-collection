# Retrieval-Augmented Generation (RAG) with ChromaDB

This example demonstrates how to create vector databases and query using [ChromaDB](https://docs.trychroma.com/).

## Prerequisites
- [Pixi](https://pixi.prefix.dev/latest/): package management
    - `pixi.toml`: package dependencies
    - `pixi.lock`: package lock file
- Python
- ChromaDB
    - Create a database on [Chroma Cloud](https://www.trychroma.com/)
    - Get API key in a selected database via "Settings/Connect to your database"
    - Set the environment variables in `.env`
        ```bash=
        CHROMA_API_KEY=your_api_key
        CHROMA_TENANT=your_tenant_id
        CHROMA_DATABASE=your_database_id
        ```
- Ollama (for chat)
    - Install [Ollama](https://ollama.com/)
    - Pull the model (available models: https://ollama.com/library)
        ```bash
        ollama pull mistral
        ```
        - Choose models with tool-calling capability

## Getting-started
- Clone the repository
    ```bash
    git clone https://github.com/tengssh/cms-collection.git
    cd cms-collection/skills/rag-chroma
    ```
- Install dependencies (optional)
    ```bash
    pixi install
    ```
- Run the tasks
    - Ingestion
        ```bash
        # Ingest the default markdown file (../../../README.md)
        pixi run ingest

        # Or specify a custom markdown file path
        pixi run ingest --file ../../README.md
        ```
        - Without an API key, a local database will be created in `./chroma_db`.
    - Search
        ```bash
        # Search using default parameters
        pixi run search

        # Search using custom query, limit, sections, or tags
        pixi run search --query "DFT software" --limit 5 --sections "Curated Lists" "Tools"
        ```
        - Available search arguments:
            - `--query`: Query text to search for (default: `"database for 2D materials"`)
            - `--limit`: Number of results to return (default: `10`)
            - `--sections`: Filter results by specific sections (default: `"Curated Lists" "Databases & Datasets" "Tools"`)
            - `--tags`: Filter results by specific tags (default: `None`)
    - Chat
        - Start the Ollama server (in one terminal)
            ```bash
            ollama serve
            ```
        - Run the chat (in another terminal)
            ```bash
            pixi run chat
            ```
        - Note that AI may hallucinate. Please verify the information.
