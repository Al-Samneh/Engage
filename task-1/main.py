import os
import sys

# Ensure local imports (agents, state, etc.) work both when run directly and via the API.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Disable telemetry for ChromaDB and other tools
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["SCARF_NO_ANALYTICS"] = "true"

import logging
# Quiet down verbose third-party loggers so the CLI stays readable.
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)

# Attempt to monkeypatch Posthog capture to silence "capture() takes 1 positional argument..." error
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    try:
        import chromadb.telemetry.posthog
        chromadb.telemetry.posthog.Posthog.capture = lambda *args, **kwargs: None
    except ImportError:
        pass

from langgraph.graph import StateGraph, END
from state import GraphState
from agents.query_agent import query_extractor_node
from agents.retrieval_agent import retriever_node
from agents.response_agent import responder_node

def build_graph():
    """
    Wire the three LangGraph nodes that make up the RAG assistant.
    Keeping this in one place makes it obvious how state flows.
    """
    workflow = StateGraph(GraphState)

    # Nodes stay small and stateless; business logic lives inside each agent file.
    workflow.add_node("extract_query", query_extractor_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", responder_node)
    
    workflow.set_entry_point("extract_query")
    workflow.add_edge("extract_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    chat_history = []
    
    print("🤖: Hello! I can help you find restaurants in UAE. (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # LangGraph expects a dict-like state. We only populate the pieces that
            # Agent 1 needs; the rest will be filled in as nodes run.
            inputs = {
                "question": user_input,
                "messages": chat_history
            }
            
            result = app.invoke(inputs)
            response = result["generation"]
            
            print(f"AI: {response}\n")
            
            # Append to history AFTER printing so conversation order is preserved.
            chat_history.append(f"User: {user_input}")
            chat_history.append(f"AI: {response}")
            
        except Exception as e:
            print(f"Error: {e}")