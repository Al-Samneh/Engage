"""
Agent 2: uses structured filters plus semantic similarity to pull the best set of
candidate restaurants from Chroma. Treats metadata as the source of truth so
downstream prompts never have to guess about price or city.
"""
from config import COLLECTION_NAME, get_embeddings, get_chroma_client
from utils import debug_log

def retriever_node(state):
    print("--- AGENT 2: RETRIEVING DATA ---")
    filters = state["filters"]
    question = state["question"]
    
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)
    
    conditions = []
    
    # LOCATION: metadata was lowercase at ingest time, so we normalize here too.
    if filters.get("location"):
        loc_filter = filters["location"].lower()
        conditions.append({"location": {"$eq": loc_filter}})
    
    # PRICE: only an upper bound is needed because menus were stored as
    # min/max pairs and most users specify a ceiling.
    if filters.get("price_max"):
        price_val = filters["price_max"]
        conditions.append({"price_max": {"$lte": price_val}})
        
    # CUISINE: equality filter keeps a "romantic" request from drifting into
    # unrelated cuisines if Agent 1 mis-parsed the text.
    if filters.get("cuisine"):
        cuisine_filter = filters["cuisine"].lower()
        conditions.append({"cuisine": {"$eq": cuisine_filter}})
    
    # the way chroma works, need a dict with an $and if there are multiple conditions that need to be true!
    where_clause = None
    if len(conditions) > 1:
        where_clause = {"$and": conditions} 
    elif len(conditions) == 1:
        where_clause = conditions[0]

    embed_model = get_embeddings()
    query_vec = embed_model.embed_query(question)
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5, 
        where=where_clause
    )
    
    docs = results['documents'][0] if results['documents'] else []
    metas = results['metadatas'][0] if results['metadatas'] else []
    
    context_list = []
    for i in range(len(docs)):
        context_list.append(f"Info: {docs[i]} | Meta: {metas[i]}")

    debug_log("2_retriever_logic", {
        "applied_filters": where_clause, 
        "semantic_query": question,
        "raw_db_results": metas,
        "final_context_list": context_list
    })
        
    return {"documents": context_list}
