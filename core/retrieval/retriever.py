"""Logic for retrieving relevant docs"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from ..utils import startup
from ..utils.logger import logger
from ..utils.helpers import timer, returns
from ..utils.config import huggingface_config, pinecone_config


# Initialize embedding model for query embedding
_embedding_config = huggingface_config()
_embedding_model = HuggingFaceEmbeddings(
    model_name=_embedding_config.get('embedding model',
                                      'sentence-transformers/all-MiniLM-L6-v2')
)

# establishing function gate
vector_db = None

# retrieves context to make the response knowledgable
@timer  # to keep track of time taken for retrieval
@returns(list)  # we expect a list of relevant documents to be returned
def retrieve_context(query: str, namespace: str = 'None', top_k: int = 7):
    """retrieves relevant context for a given query"""
    global vector_db
    # perform retrieval using the vector database
    if vector_db is None:
        vector_db = startup.get_resources()["vector_db"]
        index = vector_db.Index(pinecone_config()['index'].lower())
    if namespace == 'None':
        logger.warning(
            "No namespace provided for retrieval. Using default namespace."
        )
        raise ValueError("Namespace is required for retrieval.")

    # Step 1: Embed the query string
    try:
        query_embedding = _embedding_model.embed_query(query)
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        raise

    # Step 2: Query the vector store with the embedding
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace)
        
        return results['matches']
    except Exception as e:
        logger.error("Vector DB query failed: %s", e)
        raise
