import os
from chromadb import Search, K, Knn, Rrf
import chromadb

# --- STEP 0: Connect to Chroma ---
if os.getenv("CHROMA_API_KEY"):
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
    )
    print(">> Using CloudClient")
else:
    client = chromadb.PersistentClient(path="./chroma_db")
    print(">> Using PersistentClient")

# --- STEP 1: Get the Collection ---
collection = client.get_collection("cms_collection")

# --- STEP 2: Define the Query ---
query_text = "database for 2D materials"
target_sections = ["Curated Lists", "Databases & Datasets", "Tools"]
target_tags = None #["Data", "App"]

# --- STEP 3: Perform the Search ---
# 1. Dense Semantic Search
dense_rank = Knn(
    query=query_text, 
    key="#embedding",
    limit=200,
    return_rank=True
)

# 2. Sparse Keyword Search
sparse_rank = Knn(
    query=query_text, 
    key="sparse_embedding",
    limit=200,
    return_rank=True
)

# 3. Combine with RRF
use_search_api = True
if use_search_api and os.getenv("CHROMA_API_KEY"):
    # Search API only available in cloud
    filter = None
    if target_sections:
        filter = K("section").is_in(target_sections)
    if target_tags:
        tag_filter = K("tags").contains(target_tags[0])
        for tag in target_tags[1:]:
            tag_filter |= K("tags").contains(tag)
        if filter is not None:
            filter = filter & tag_filter

    search = (
        Search()
        .where(filter)
        .rank(Rrf(ranks=[dense_rank, sparse_rank], weights=[0.7, 0.3])) # hybrid search
        .limit(10)
        .select(K.DOCUMENT, K.SCORE, "section", "url", "tags")
    )
    results = collection.search(search)
else:
    query_args = {
        "query_texts": [query_text],
        "n_results": 10,
        "include": ["documents", "metadatas", "distances"]
    }
    if target_sections:
        query_args["where"] = {"section": {"$in": target_sections}}
    if target_tags:
        query_args["where_document"] = {"$or": [{"$contains": tag} for tag in target_tags]}
    results = collection.query(**query_args)

# --- STEP 4: Display the Results ---
ids = results.get('ids', [[]])[0]
for i in range(len(ids)):
    raw_document = results.get('documents', [[]])[0][i]
    metadata = results.get('metadatas', [[]])[0][i]
    if os.getenv("CHROMA_API_KEY"):
        score_name = 'RRF Score'
        score = results.get('scores', [[]])[0][i]
    else:
        score_name = 'Distance'
        score = results.get('distances', [[]])[0][i]

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