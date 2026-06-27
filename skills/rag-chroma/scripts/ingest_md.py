import os
import uuid
import argparse
from chroma_utils import (
    get_chroma_client,
    get_or_create_collection,
    parse_markdown as _parse_markdown,
)

db_sections = [
    "Databases & Datasets",
    "Curated Lists",
    "Computing & Workflows",
    "Machine Learning",
    "Tools: Crystal structures",
    "Tools: Molecular structures",
    "Toolkits",
    "OCW"
]

def parse_markdown(text):
    """
    Wrapper of parse_markdown from chroma_utils for backwards compatibility with tests.
    """
    return _parse_markdown(text, db_sections=db_sections)

def run_ingestion(text, collection=None):
    """
    Ingests parsed markdown document chunks into Chroma DB.
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        
    chunks = parse_markdown(text)
    if not chunks:
        print("No chunks found to ingest.")
        return
        
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks], 
        documents=[chunk.page_content for chunk in chunks], 
        metadatas=[chunk.metadata for chunk in chunks]
    )
    print(f">> Successfully ingested {len(chunks)} chunks.")

def main():
    parser = argparse.ArgumentParser(description="Ingest markdown file into Chroma DB.")
    parser.add_argument(
        "--file", 
        type=str, 
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../README.md"),
        help="Path to the markdown file to ingest."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        return

    with open(args.file, 'r', encoding='utf-8') as f:
        run_ingestion(f.read())

if __name__ == "__main__":
    main()