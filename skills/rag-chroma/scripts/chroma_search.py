import argparse
from chroma_utils import (
    get_chroma_client,
    get_collection,
    search_collection,
)

def run_search(query_text, limit=10, target_sections=None, target_tags=None, collection=None):
    """
    Executes search and prints formatted results to console.
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_collection(client)
        
    results = search_collection(
        collection=collection,
        query_text=query_text,
        limit=limit,
        target_sections=target_sections,
        target_tags=target_tags
    )
    
    ids = results["ids"]
    score_name = results["score_name"]
    
    for i in range(len(ids)):
        raw_document = results["documents"][i]
        metadata = results["metadatas"][i]
        score = results["scores"][i]
        
        url = metadata.get("url", "No URL provided")
        tags = metadata.get("tags", "No tags")

        doc_split = raw_document.split('|')
        section = doc_split[0].replace("Section:", "").strip() if len(doc_split) > 0 else "N/A"
        resource = doc_split[1].replace("Resource:", "").strip() if len(doc_split) > 1 else "N/A"
        description = doc_split[2].replace("Description:", "").strip() if len(doc_split) > 2 else ""

        print(f"Result #{i+1}")
        print(f"{score_name}: {score:.4f}")
        print(f"Resource: {resource}")
        print(f"URL: {url}")
        print(f"Description: {description[:120]}...")
        print(f"Tags: {tags}")
        print(f"Section: {section}")
        print("-" * 3)

def main():
    parser = argparse.ArgumentParser(description="Query Chroma DB collection.")
    parser.add_argument(
        "--query", 
        type=str, 
        default="database for 2D materials",
        help="Query text to search for."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Number of results to return."
    )
    parser.add_argument(
        "--sections", 
        type=str, 
        nargs="*",
        default=["Curated Lists", "Databases & Datasets", "Tools"],
        help="Filter results by specific sections."
    )
    parser.add_argument(
        "--tags", 
        type=str, 
        nargs="*",
        default=None,
        help="Filter results by specific tags."
    )
    
    args = parser.parse_args()
    
    run_search(
        query_text=args.query,
        limit=args.limit,
        target_sections=args.sections,
        target_tags=args.tags
    )

if __name__ == "__main__":
    main()