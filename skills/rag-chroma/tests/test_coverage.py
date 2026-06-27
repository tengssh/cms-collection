import os
import pytest
from unittest.mock import MagicMock, patch

from ollama_chat import unwrap_to_string
from chroma_utils import parse_markdown, search_collection, get_workspace_root

# ==========================================
# 1. Tests for unwrap_to_string (Query Parsing)
# ==========================================

def test_unwrap_to_string_plain():
    assert unwrap_to_string("simple query") == "simple query"

def test_unwrap_to_string_simple_dict():
    assert unwrap_to_string({"query": "materials science"}) == "materials science"
    assert unwrap_to_string({"type": "dft simulation"}) == "dft simulation"

def test_unwrap_to_string_nested_dict():
    assert unwrap_to_string({"query": {"type": "uncertainty quantification"}}) == "uncertainty quantification"
    assert unwrap_to_string({"query": {"query": "molecular dynamic"}}) == "molecular dynamic"

def test_unwrap_to_string_list():
    assert unwrap_to_string(["first query", "second query"]) == "first query"
    assert unwrap_to_string([{"query": "nested in list"}]) == "nested in list"

def test_unwrap_to_string_fallback():
    # If standard keys aren't present, fall back to the first value in dict
    assert unwrap_to_string({"unknown_key": "fallback value"}) == "fallback value"
    assert unwrap_to_string({}) == ""

# ==========================================
# 2. Tests for Recursive Markdown Parsing
# ==========================================

def test_recursive_markdown_parsing():
    workspace_root = get_workspace_root()
    temp_dir = os.path.join(workspace_root, "docs")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_name = "test_temp_child.md"
    temp_file_path = os.path.join(temp_dir, temp_file_name)
    
    child_content = """---
name: Temp Child Topic
description: Temp description
tags: ["TempTag"]
---
## Temp Child Section

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [TempItem](https://example.com) | Temp item description. | TempTag |
"""
    
    # Write temp child file
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(child_content)
        
    try:
        parent_markdown = """## Databases & Datasets

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [SubFile](./docs/test_temp_child.md) | Parent link. | Curated |
"""
        chunks = parse_markdown(parent_markdown)
        
        # Check that we parsed the parent row AND the child resource
        assert len(chunks) == 2
        
        # Parent chunk check
        parent_doc = chunks[0]
        assert parent_doc.metadata["file"] == "README.md"
        assert "SubFile" in parent_doc.page_content
        
        # Child chunk check
        child_doc = chunks[1]
        assert child_doc.metadata["file"] == "docs/test_temp_child.md"
        assert child_doc.metadata["section"] == "Databases & Datasets"
        assert child_doc.metadata["sub_section"] == "Temp Child Topic"
        assert child_doc.metadata["url"] == "https://example.com"
        assert "TempItem" in child_doc.page_content
        assert "Temp item description" in child_doc.page_content
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ==========================================
# 3. Tests for search_collection Formatting
# ==========================================

def test_search_collection_local():
    mock_collection = MagicMock()
    # Mock return value of query() for local Chroma
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"tag": "t1"}, {"tag": "t2"}]],
        "distances": [[0.123, 0.456]]
    }
    
    # Ensure CHROMA_API_KEY is not in env during this test
    with patch.dict(os.environ, {}, clear=True):
        res = search_collection(mock_collection, query_text="test query", limit=2)
        
        assert res["score_name"] == "Distance"
        assert res["ids"] == ["id1", "id2"]
        assert res["scores"] == [0.123, 0.456]
        mock_collection.query.assert_called_once()

def test_search_collection_cloud():
    mock_collection = MagicMock()
    # Mock return value of search() for cloud Chroma
    mock_collection.search.return_value = {
        "ids": [["cloud_id1"]],
        "documents": [["cloud_doc1"]],
        "metadatas": [[{"tag": "cloud_t1"}]],
        "scores": [[0.95]]
    }
    
    # Patch environment to simulate Chroma Cloud
    with patch.dict(os.environ, {"CHROMA_API_KEY": "dummy_api_key"}):
        res = search_collection(mock_collection, query_text="test query", limit=1)
        
        assert res["score_name"] == "RRF Score"
        assert res["ids"] == ["cloud_id1"]
        assert res["scores"] == [0.95]
        mock_collection.search.assert_called_once()
