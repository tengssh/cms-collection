import sys
from unittest.mock import patch
import pytest

def test_no_side_effects_on_import():
    # Clear target modules from sys.modules to force a re-import and trigger top-level code execution.
    for mod in ["ingest_md", "chroma_search", "ollama_chat", "chroma_utils"]:
        sys.modules.pop(mod, None)
        
    with patch("chromadb.PersistentClient") as mock_persistent, \
         patch("chromadb.CloudClient") as mock_cloud, \
         patch("langchain_ollama.ChatOllama") as mock_chat_ollama, \
         patch("langchain.agents.create_agent") as mock_create_agent:
         
         # Force import
         import ingest_md
         import chroma_search
         import ollama_chat
         
         # In the refactored code, these should not be called on import.
         # In the old code, they will be called, making this test fail.
         mock_persistent.assert_not_called()
         mock_cloud.assert_not_called()
         mock_chat_ollama.assert_not_called()
         mock_create_agent.assert_not_called()

def test_chroma_utils_existence():
    # Verify chroma_utils exists and has the expected functions.
    # This will fail on import error initially since chroma_utils.py doesn't exist yet.
    import chroma_utils
    assert hasattr(chroma_utils, "get_chroma_client")
    assert hasattr(chroma_utils, "get_or_create_collection")
    assert hasattr(chroma_utils, "get_collection")
    assert hasattr(chroma_utils, "search_collection")
    assert hasattr(chroma_utils, "parse_markdown")
