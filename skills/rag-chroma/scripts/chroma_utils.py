import os
import re
import uuid
from dotenv import load_dotenv
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def get_chroma_client():
    """
    Connects to Chroma and returns client.
    Uses CloudClient if CHROMA_API_KEY is present in environment,
    otherwise uses PersistentClient.
    """
    if os.getenv("CHROMA_API_KEY"):
        print(">> Using CloudClient")
        return chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
    else:
        print(">> Using PersistentClient")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "chroma_db")
        return chromadb.PersistentClient(path=db_path)

def get_or_create_collection(client, name="cms_collection"):
    """
    Gets or creates a Chroma collection with the defined schema.
    """
    from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, ChromaBm25EmbeddingFunction
    
    is_cloud = bool(os.getenv("CHROMA_API_KEY"))
    source_key = K.DOCUMENT if is_cloud else "#document"
    
    schema = Schema()
    schema.create_index(
        config=VectorIndexConfig(
            source_key=source_key,
            embedding_function=DefaultEmbeddingFunction()
        )
    )
    
    if is_cloud:
        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=source_key,
                embedding_function=ChromaBm25EmbeddingFunction(),
                bm25=True
            ),
            key="sparse_embedding"
        )
        
    return client.get_or_create_collection(name=name, schema=schema)

def get_collection(client, name="cms_collection"):
    """
    Gets an existing Chroma collection.
    """
    return client.get_collection(name=name)

def parse_markdown(text, db_sections=None):
    """
    Parses a markdown document into Document chunks based on section headers and tables.
    """
    if db_sections is None:
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
        
    headers_to_split_on = [("#", "Title"), ("##", "Section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = splitter.split_text(text)
    
    n_lines = 1
    chunks = []
    for i_section, section in enumerate(sections):
        section_title = section.metadata.get("Section")
        content = section.page_content
        
        lines = content.split('\n')
        if section_title not in db_sections:
            n_lines += len(lines) + 1
            continue

        for i_line, line in enumerate(lines):
            if '|' in line and not any(x in line for x in ['| :---', '| Item']):
                cols = [c.strip() for c in line.split('|') if c.strip()]
                
                if len(cols) >= 2:
                    item_raw = cols[0]
                    desc = cols[1]
                    tags = cols[2] if len(cols) > 2 else ""
                    
                    name_match = re.search(r'\[(.*?)\]', item_raw)
                    url_match = re.search(r'\]\((.*?)\)', item_raw)
                    
                    name = name_match.group(1) if name_match else item_raw
                    url = url_match.group(1) if url_match else ""

                    doc = Document(
                        page_content=f"Section: {section_title} | Resource: {name} | Description: {desc}",
                        metadata={
                            "line": (i_line + 1) + (i_section + 1) + n_lines,
                            "section": section_title,
                            "url": url,
                            "tags": tags,
                            "type": "table_row"
                        }
                    )
                    chunks.append(doc)
            else: # for debug
                doc = Document(
                    page_content=f"Section: {section_title} | Resource: {line} | Description: None",
                    metadata={
                        "line": (i_line + 1) + (i_section + 1) + n_lines,
                        "section": section_title,
                        "url": "",
                        "tags": "",
                        "type": "table_row"
                    }
                )
        n_lines += len(lines) + 1
    return chunks

def search_collection(collection, query_text, limit=10, target_sections=None, target_tags=None):
    """
    Performs search on Chroma DB collection.
    Supports Cloud RRF hybrid search if CHROMA_API_KEY is present,
    otherwise uses local vector distance query.
    Returns a standardized dictionary of results:
    {
        "ids": [...],
        "documents": [...],
        "metadatas": [...],
        "scores": [...],      # RRF score or Distance
        "score_name": str     # "RRF Score" or "Distance"
    }
    """
    from chromadb import Search, K, Knn, Rrf

    is_cloud = bool(os.getenv("CHROMA_API_KEY"))

    if is_cloud:
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

        filter_expr = None
        if target_sections:
            filter_expr = K("section").is_in(target_sections)
        if target_tags:
            tag_filter = K("tags").contains(target_tags[0])
            for tag in target_tags[1:]:
                tag_filter |= K("tags").contains(tag)
            if filter_expr is not None:
                filter_expr = filter_expr & tag_filter
            else:
                filter_expr = tag_filter

        search = (
            Search()
            .where(filter_expr)
            .rank(Rrf(ranks=[dense_rank, sparse_rank], weights=[0.7, 0.3]))
            .limit(limit)
            .select(K.DOCUMENT, K.SCORE, "section", "url", "tags")
        )
        results = collection.search(search)
        
        # Standardize cloud results
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        scores = results.get('scores', [[]])[0]
        score_name = "RRF Score"
    else:
        query_args = {
            "query_texts": [query_text],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"]
        }
        # Local persistent client queries
        if target_sections:
            query_args["where"] = {"section": {"$in": target_sections}}
        if target_tags:
            if len(target_tags) == 1:
                query_args["where_document"] = {"$contains": target_tags[0]}
            else:
                query_args["where_document"] = {"$or": [{"$contains": tag} for tag in target_tags]}
        results = collection.query(**query_args)
        
        # Standardize local results
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        scores = results.get('distances', [[]])[0]
        score_name = "Distance"

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "scores": scores,
        "score_name": score_name
    }
