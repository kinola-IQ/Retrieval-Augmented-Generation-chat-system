"""handles startup and shutdown events for the application"""

from pinecone import Pinecone
from .config import pinecone_config
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt
)

# database which holds all the answers
VECTOR_DB = None
_CONFIGURED = False


@retry(wait=wait_random_exponential(10, 40), stop=stop_after_attempt(5))
async def make_connections():
    """initializes resources on startup"""
    global VECTOR_DB
    # Initialize resources here (e.g., database connections, API clients)
    if pinecone_config()["api_key"] is None:
        raise ValueError(
            "Pinecone API key is not set in environment variables."
            )
    VECTOR_DB = Pinecone(api_key=pinecone_config()[0])

    # status to inform us of successful connection on all parts
    _CONFIGURED = True


def get_resources():
    """returns initialized resources"""
    # we want access to resource needed for the system to work,
    # so we check if they are initialized before returning
    if VECTOR_DB is None:
        raise ValueError(
            "Vector database connection has not been established."
            )
    return VECTOR_DB

def connection_success():
    """informs us of successful connection of all resources"""
    if _CONFIGURED is False:
        raise ValueError(
            "resources not configured yet"
            )
    return _CONFIGURED
