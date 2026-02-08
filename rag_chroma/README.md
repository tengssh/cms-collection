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
    cd cms-collection/rag_chroma
    ```
- Install dependencies (optional)
    ```bash
    pixi install
    ```
- Run the tasks
    - Ingestion
        ```bash
        pixi run ingest
        ```
        - Without an API key, a local database will be created in `./chroma_db`.
    - Search
        ```bash
        pixi run search
        ```
    - Chat with the database
        - Start the Ollama server (in one terminal)
            ```bash
            ollama serve
            ```
        - Run the chat (in another terminal)
            ```bash
            pixi run chat
            ```
        - Note that AI may hallucinate. Please verify the information.
