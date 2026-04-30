"""Logic for retrieving relevant docs"""

from ..utils.startup import get_resources
from ..utils.logger import logger
from ..utils.helpers import timer, returns


# load resources needed for retrieval
resources = get_resources()

# vector_db is needed for retrieval,
# so we make it available at the module level
vector_db = resources["vector_db"]


# retrieves context to make the response knowledgable
@timer # to keep track of time taken for retrieval
@returns(list) # we expect a list of relevant documents to be returned
def retrieve_context(query: str,  namespace: str = 'None', top_k: int = 5,):
    """retrieves relevant context for a given query"""
    # perform retrieval using the vector database
    if namespace is 'None':
        logger.warning(
            "No namespace provided for retrieval. Using default namespace."
        )
        raise ValueError("Namespace is required for retrieval.")
    results = vector_db.query(
        query,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace)
    return results
