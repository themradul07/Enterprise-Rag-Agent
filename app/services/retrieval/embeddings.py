import logfire
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 50
EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    """Initialise SentenceTransformer embedding model once per process."""
    global _model
    if _model is None:
        logfire.info("Loading sentence-transformers embedding model (all-mpnet-base-v2, 768-dim).")
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model

def get_embedding_dim() -> int:
    """Return the vector dimension for the active embedding model."""
    return EMBEDDING_DIM

# Backward-compatibility alias
get_embeddings_model = get_embedding_dim

def embed_query(query: str) -> list[float]:
    """Generate a 768-dim embedding vector for a single query."""
    model = _get_model()
    with logfire.span("Embed query", model="all-mpnet-base-v2"):
        embedding = model.encode([query], show_progress_bar=False)[0]
        return embedding.tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch encodes multiple text chunks in batches of BATCH_SIZE."""
    model = _get_model()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("Embed batch", model="all-mpnet-base-v2", start=i, size=len(batch)):
            batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
            all_embeddings.extend(batch_embeddings)
    return all_embeddings