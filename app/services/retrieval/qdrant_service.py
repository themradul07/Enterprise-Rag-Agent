import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embeddings import embed_query

# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
) 

def search_enterprise_knowledge(query:str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        print(f"\n🚀 [VECTOR DB - QDRANT] Generating embedding vector for query: '{query}'")
        query_vector = embed_query(query)

        print(f"🎯 [VECTOR DB - QDRANT] Querying collection '{settings.QDRANT_COLLECTION}' (limit={limit})...")
        # Using query_points - the mordern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        results = []
        for res in response.points:
            results.append({
                "content":res.payload.get("text",""),
                "source":res.payload.get("source", "Unknown"),
                "score":res.score
            })
        print(f"✅ [VECTOR DB - QDRANT] Found {len(results)} matching chunks in Qdrant (top score: {results[0]['score'] if results else 'N/A'})\n")
        return results
    except Exception as e:
        print(f"❌ [VECTOR DB - QDRANT] Search Failed: {e}")
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []