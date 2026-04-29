"""Logic for retrieving relevant docs"""

from ..utils.startup import get_resources

# load resources needed for retrieval
resources = get_resources()

# vector_db is needed for retrieval,
# so we make it available at the module level
vector_db = resources["vector_db"]

def retrieve_context(query: str, top_k: int = 5):
    """retrieves relevant context for a given query"""
    # perform retrieval using the vector database
    results = vector_db.query(query, top_k=top_k)
    return results