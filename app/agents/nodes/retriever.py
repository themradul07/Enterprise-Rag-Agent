import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    print("\n" + "="*60)
    print(f"🔍 [RETRIEVER NODE INVOKED]")
    print(f"   Target Search Query: '{query}'")
    print("="*60)

    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        print(f"📦 [RETRIEVER] Fetched {len(raw_results)} candidate chunks from Vector DB.")
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        with logfire.span("⚖️ Semantic Reranking"):
            print("⚖️ [RERANKER] Running FlashRank cross-encoder reranking...")
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            print(f"✅ [RERANKER] Selected top {len(reranked_contents)} highest-relevance chunks for context.")
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
    
    print("="*60 + "\n")
    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }