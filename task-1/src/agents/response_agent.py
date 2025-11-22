"""
Agent 3: turns the retrieved snippets into a concise reply while enforcing the
“no hallucinations” rule. All personalization must be grounded in the provided
context, so the prompt leans heavily on the documents plus chat history.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm
from utils import debug_log

def responder_node(state):
    print("--- AGENT 3: GENERATING RESPONSE (GEMINI) ---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("messages", [])
    
    llm = get_llm()
    
    # Placeholder for future fallback tagging logic.
    is_fallback = False
    if documents and "[NOTE:" in documents[0]:
        is_fallback = True
        
    context_str = "\n\n".join(documents)
    
    template = """You are an elite restaurant concierge.
    
    YOUR GOAL:
    Provide a personalized, engaging recommendation based strictly on the provided context.
    
    CONDITIONAL LOGIC:
    1. If the user's specific request wasn't found (indicated by "NOTE: Semantically similar" in context), explicitly apologize and explain why you are suggesting these alternatives.
    2. If NO documents are provided, ask clarifying questions to help the user refine their search.
    3. Use the CHAT HISTORY to understand context. If the user asks "Where is it?", refer to the restaurant recommended in the previous turn.
    4. Do not provide a restaurants that is not in the database
    
    STYLE GUIDE:
    - Be warm .
    - Highlight specific dishes mentions in the context.
    - Mention the "Vibe" or amenities if relevant to the user's query.
    - IMPORTANT: BE AS CONCISE AS POSSIBLE.
    - IMPORTANT: DO NOT USE MARKDOWN FORMATTING.
    
    CHAT HISTORY:
    {chat_history}
    
    CURRENT CONTEXT (Retrieved Data):
    {context}
    
    USER'S LATEST QUESTION: 
    {question}
    
    YOUR RESPONSE:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # If we have absolutely no context to work with, nudge the user to refine the request.
    if not documents and not chat_history:
        response = "I couldn't find any restaurants matching those exact criteria. Could you perhaps broaden your search? For example, are you open to other cuisines nearby?"
    else:
        response = chain.invoke({
            "chat_history": "\n".join(chat_history),
            "context": context_str,
            "question": question
        })
    
    debug_log("3_responder_input", {
        "documents_found": len(documents),
        "is_fallback_mode": is_fallback,
        "history_length": len(chat_history)
    })
    
    return {"generation": response}
