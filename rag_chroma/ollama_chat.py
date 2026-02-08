import os
from dotenv import load_dotenv
import chromadb
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from halo import Halo

load_dotenv()

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
collection = client.get_collection(name="cms_collection")

# --- STEP 2: Define tool ---
@tool
def cms_collection_search(query: str):
    """Use this for ANY technical query."""
    from chromadb import Search, K, Knn, Rrf

    if os.getenv("CHROMA_API_KEY"):
        dense = Knn(query=query, key="#embedding", limit=20)
        sparse = Knn(query=query, key="sparse_embedding", limit=20)
        search = (
            Search()
            .rank(Rrf(ranks=[dense, sparse], weights=[0.7, 0.3])) # hybrid search
            .limit(3)
            .select(K.DOCUMENT, K.SCORE, "section", "url", "tags")
        )
        results = collection.search(search)
    else:
        query_args = {
            "query_texts": [query],
            "n_results": 3,
            "include": ["documents", "metadatas", "distances"]
        }
        results = collection.query(**query_args)

    # Parse the results
    ids = results.get('ids', [[]])[0]
    output = []
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
        output.append(
            f"**{resource}**\nSection: {section}\nDescription: {description}\nURL: {url}\nTags: {tags}\n{score_name}: {score}"
        )
    return "\n\n---\n\n".join(output) if output else "No results found."

# --- STEP 2: Create agent ---
llm = ChatOllama(model="mistral", temperature=0, num_ctx=8192)
memory = MemorySaver()

system_prompt = """You are a Research Assistant.

- For greetings: Reply with one sentence and wait for a question.
- For technical questions: You MUST use the 'cms_collection_search' tool. 
- Do not attempt to answer technical questions from your own knowledge, instead say "No information is found in the search results."
- When receiving data from 'cms_collection_search', you MUST format the result as a list under the header '## Search Results'.
"""

agent_executor = create_agent(
    model=llm, 
    tools=[cms_collection_search], 
    checkpointer=memory,
    system_prompt=system_prompt
)

# --- STEP 3: Start chat ---
def start_chat():
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
        final_msg = ""
        try:
            for chunk in agent_executor.stream(input_msg, config, stream_mode="values"):
                final_msg_obj = chunk["messages"][-1]
            spinner.stop()
            if final_msg_obj.type == "ai":
                print(f"\nAssistant: {final_msg_obj.content}\n")
        except Exception as e:
            spinner.fail("Something went wrong.")
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()