from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """
    Shared schema passed between LangGraph nodes. Keeping the contract explicit
    makes it easy to trace which agent is responsible for each key.
    """
    question: str
    filters: Dict[str, Any]
    documents: List[str]
    generation: str
    messages: List[str]