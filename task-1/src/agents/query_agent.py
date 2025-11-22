"""
Agent 1: turns a natural-language request into structured filters that downstream
retrieval can trust. Doing this up front keeps vector search focused and avoids
guesswork around price or neighborhood names.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import get_llm
from utils import debug_log  

def query_extractor_node(state):
    print("--- AGENT 1: EXTRACTING FILTERS ---")
    question = state["question"]
    messages = state.get("messages", [])
    
    # Carry over only the latest turns; older context rarely changes extraction
    # but can increase token usage.
    history_str = "\n".join(messages[-6:]) if messages else "No previous conversation."
    
    llm = get_llm()
    
    # The prompt ill pass into the LLM to extract the filters , best so far.
    prompt = PromptTemplate(
        template="""You are an expert at extracting search filters for a restaurant database.
        
        Valid Cuisines: ["Chinese", "Emirati", "French", "Indian", "Seafood", "Mexican", "Italian", "Thai", "Mediterranean", "Iranian"]
        Valid Locations: ["Al Barsha", "Downtown Dubai", "Sharjah", "Abu Dhabi", "Dubai Marina", "Business Bay", "Palm Jumeirah", "Ajman", "Jumeirah Lakes Towers (JLT)", "Jumeirah Beach Residence (JBR)"]
        
        Return a JSON object with these keys: 'location' (str|null), 'price_max' (int|null), 'cuisine' (str|null), 'amenities' (str|null).
        
        IMPORTANT RULES:
        - "cuisine" MUST be one of the Valid Cuisines list exactly. If the user asks for "romantic", "cheap", or other adjectives, do NOT put them in "cuisine".
        - "location" MUST be one of the Valid Locations list exactly. 
          - If the user says "Downtown", infer "Downtown Dubai".
          - If the user says "Marina", infer "Dubai Marina".
          - If the user says "JLT", infer "Jumeirah Lakes Towers (JLT)".
          - If the user says "JBR", infer "Jumeirah Beach Residence (JBR)".
        - "price_max" should be an Integer.
          - If user says "cheap" or "won't break the bank", use 50 - 100.
          - If user says "medium" or "moderate", use 100 - 150.
          - If user says "high" or "expensive", use 150 - 200.
          - If user says "luxury" or "fancy", use 200 - 300+.

        CONTEXT AWARENESS:
        - Use the Conversation History to inherit filters from previous turns.
        - If the User Query adds a constraint (e.g. "what about cheaper?"), keep previous constraints (cuisine, location) and update price.
        - If the User Query changes a constraint (e.g. "actually in Marina"), update location and keep others.
        - **CRITICAL EXCEPTION**: If the user says "any cuisine", "anywhere", "start over", or asks a completely new question unrelated to the previous one (like changing the city or asking for "expensive" generally), CLEAR the inherited filters.
        - **BROADENING**: If the user asks for "restaurants in Dubai" (broad city) without specifying a neighborhood, CLEAR the specific 'location' filter (e.g. remove 'Downtown Dubai').
        - **SPECIFIC RESET**: If the user asks for a new query that conflicts with old filters (e.g. "Give me expensive restaurants in Dubai" when the previous was "Italian in Sharjah"), RESET ALL filters.
        - **FRESH QUERY SAFETY**: If the current user request does NOT explicitly mention a cuisine, set "cuisine" to null even if older turns referenced one.
        - Return the FULL set of filters to apply.
        
        Conversation History:
        {chat_history}
        
        User Query: {question}
        
        JSON Output:""",
        input_variables=["question", "chat_history"]
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        filters = chain.invoke({
            "question": question,
            "chat_history": history_str
        })
        if "cuisine" in filters:
            if not question.lower().strip():
                filters["cuisine"] = None
            else:
                mentioned = any(
                    cuisine.lower() in question.lower()
                    for cuisine in ["chinese", "emirati", "french", "indian", "seafood", "mexican", "italian", "thai", "mediterranean", "iranian"]
                )
                if not mentioned:
                    filters["cuisine"] = None
    except Exception as e:
        # If parsing fails, fall back to an empty filter set so retrieval can still
        # perform a pure semantic search instead of crashing the flow.
        print(f"JSON Parsing Error: {e}")
        filters = {}
    
    debug_log("1_extractor_output", {
        "input_question": question,
        "extracted_filters": filters
    })
        
    return {"filters": filters}
