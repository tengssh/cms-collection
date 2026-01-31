import os, re
import uuid
import chromadb
from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, ChromaBm25EmbeddingFunction
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# --- STEP 0: Connect to Chroma ---
if os.getenv("CHROMA_API_KEY"):
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
    )
    source_key=K.DOCUMENT
    print(">> Using CloudClient")
else:
    client = chromadb.PersistentClient(path="./chroma_db")
    source_key="#document"
    print(">> Using PersistentClient")

# --- STEP 1: Define the Schema ---
schema = Schema()
schema.create_index(
    config=VectorIndexConfig(
        source_key=source_key,
        embedding_function=DefaultEmbeddingFunction()
    )
)

if os.getenv("CHROMA_API_KEY"): # Sparse index only available in cloud
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=source_key,
            embedding_function=ChromaBm25EmbeddingFunction(),
            bm25=True
        ),
        key="sparse_embedding"
    )

collection = client.get_or_create_collection(name="cms_collection", schema=schema)

# --- STEP 2: Use LangChain to Parse ---
db_sections = ["Databases & Datasets", "Curated Lists"]

def parse_markdown(text):
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
            else:
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
                chunks.append(doc)
        n_lines += len(lines) + 1
    return chunks

# --- STEP 3: Upload ---
def run_ingestion(text):
    chunks = parse_markdown(text)
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks], 
        documents=[chunk.page_content for chunk in chunks], 
        metadatas=[chunk.metadata for chunk in chunks]
    )

if __name__ == "__main__":
    with open('../README.md', 'r', encoding='utf-8') as f:
        run_ingestion(f.read())