1. Bias for Action (Build vs. Plan Trade-offs)
Infrastructure Choice: Instead of spending days properly setting up a cloud vector database (like Pinecone) or complex SQL schemas, I chose ChromaDB (Local). This allowed me to start building the retrieval logic immediately without infrastructure overhead.
Model Selection: I selected gemini-2.5-flash-lite over larger, slower models. I prioritized latency and user experience (sub-second search) over theoretical maximum intelligence, recognizing that a search agent needs speed more than deep reasoning.

2. Ownership (Decision Validation)
Validation System: I didn't just trust the LLM. I built a custom debug_log system (saving intermediate JSONs to debug/) to verify exactly what filters Agent 1 was extracting and what documents Agent 2 was retrieving. This allowed me to catch and fix issues objectively.
Data Cleaning: When I saw the price data was messy ("AED 150 - 200"), I wrote a robust process_price function to parse strings into integers, enabling real mathematical filtering ($lte) instead of unreliable text matching.
Hallucination Fix: When I noticed the agent inventing restaurants (like "Ossiano") for expensive queries, I took ownership of the quality. I implemented Negative Constraints in the prompt ("Do not use internal knowledge") to force the agent to admit when the database was empty, prioritizing truthfulness over "being helpful."

3. Innovation (Novel Approaches)
Metadata-First Retrieval: A standard RAG system relies on vector similarity, which fails at hard numbers (prices) or specific locations. I innovated by building a "Code-First" Filter Extractor (Agent 1) that translates natural language into structured metadata queries before doing semantic search. This guarantees precision (filtering) + recall (vibes).
Smart Context Reset: I went beyond simple chat history. I implemented a "Smart Reset" logic where the agent detects if the user changes topic (e.g., "Start over" or switching cities) and proactively clears old filters. This solves the common RAG problem of "sticky context" where old constraints ruin new searches.