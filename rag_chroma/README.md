# Retrieval-Augmented Generation (RAG) with ChromaDB

This example demonstrates how to create vector databases and query using [ChromaDB](https://docs.trychroma.com/).

## Prerequisites
- [Pixi](https://pixi.prefix.dev/latest/): package management
    - `pixi.toml`: package dependencies
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

## Getting-started
- Clone the repository
    ```bash
    git clone https://github.com/tengssh/cms-collection.git
    cd cms-collection/rag_chroma
    ```
- Initialize the environment
    ```bash
    pixi init
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
    - Chat with the database (WIP)