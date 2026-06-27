import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from halo import Halo

from chroma_utils import (
    get_chroma_client,
    get_collection,
    search_collection,
)

# Global variables for lazy initialization
_collection = None
_agent_executor = None

def get_shared_collection():
    """
    Lazily initializes and retrieves the Chroma DB collection instance.
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = get_collection(client, name="cms_collection")
    return _collection

from typing import Union, Any

def unwrap_to_string(val: Any) -> str:
    """
    Recursively unwraps dictionaries and lists to extract a plain string.
    """
    if isinstance(val, dict):
        for key in ["query", "type", "description", "content"]:
            if key in val:
                unwrapped = unwrap_to_string(val[key])
                if unwrapped:
                    return unwrapped
        for v in val.values():
            unwrapped = unwrap_to_string(v)
            if unwrapped:
                return unwrapped
        return ""
    elif isinstance(val, list):
        for item in val:
            unwrapped = unwrap_to_string(item)
            if unwrapped:
                return unwrapped
        return ""
    return str(val)

@tool
def cms_collection_search(query: Union[str, dict]):
    """Use this for ANY technical query."""
    query_str = unwrap_to_string(query)
    collection = get_shared_collection()
    results = search_collection(collection, query_text=query_str, limit=3)

    ids = results["ids"]
    score_name = results["score_name"]
    output = []
    
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
        
        output.append(
            f"**{resource}**\nSection: {section}\nDescription: {description}\nURL: {url}\nTags: {tags}\n{score_name}: {score}"
        )
    return "\n\n---\n\n".join(output) if output else "No results found."

def get_agent_executor():
    """
    Lazily constructs the ChatOllama agent executor.
    """
    global _agent_executor
    if _agent_executor is None:
        llm = ChatOllama(model="llama3.2:3b", temperature=0, num_ctx=8192)
        memory = MemorySaver()

        system_prompt = """You are a Research Assistant.

- For greetings: Reply with one sentence and wait for a question.
- For technical questions: You MUST use the 'cms_collection_search' tool. Pass a simple, plain text search string (e.g. keywords) as the query argument (do NOT pass structured objects, JSON, or nested dictionaries).
- Do not attempt to answer technical questions from your own knowledge, instead say "No information is found in the search results."
- When receiving data from 'cms_collection_search', you MUST format the result as a list under the header '## Search Results'.
"""
        _agent_executor = create_agent(
            model=llm, 
            tools=[cms_collection_search], 
            checkpointer=memory,
            system_prompt=system_prompt
        )
    return _agent_executor

def start_chat():
    agent_executor = get_agent_executor()
    config = {"configurable": {"thread_id": "repl_session_1"}}
    
    print("\n" + "="*42)
    print("  RAG Chat Agent (Ollama + Chroma)")
    print("  Type 'exit' to quit | 'clear' to reset")
    print("="*42 + "\n")

    while True:
        user_input = input(">>> ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            config["configurable"]["thread_id"] += "_new"
            print("System: Memory cleared.\n")
            continue
        if not user_input:
            continue
        
        # Initialize the spinner
        spinner = Halo(text='Thinking...', spinner='dots')
        spinner.start()

        # Stream the agent's message
        input_msg = {"messages": [("user", user_input)]}
        try:
            final_msg_obj = None
            for chunk in agent_executor.stream(input_msg, config, stream_mode="values"):
                final_msg_obj = chunk["messages"][-1]
            spinner.stop()
            if final_msg_obj and final_msg_obj.type == "ai":
                print(f"\nAssistant: {final_msg_obj.content}\n")
        except Exception as e:
            spinner.fail("Something went wrong.")
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()
