"""Script to preprocess & index documents"""

import itertools
import uuid
import numpy as np

# custom modules
from ..core.utils.config import pinecone_config
from ..core.utils.startup import get_resources
from ..core.retrieval.embeddings import (
    load_and_split_document, create_embeddings)

# load resources needed for retrieval
resources = get_resources()
pinecone_config = pinecone_config()

# vector_db is needed for injesting,
# so we make it available at the module level
vector_db_client = resources["vector_db"]
namespace, index_name = pinecone_config["namespace", 'index']


# turning vectors to iterables to reduce memory cost
def chunks(file, batch_size: int = 100):
    """serves data on demand streams to conserve memory usage"""
    # making the doc an iterable
    doc = load_and_split_document(file)
    iterable = iter(doc)

    # we are slicing the document into batches
    # and yielding its content one at a time to reduce memory usage
    chunk = tuple(itertools.islice(iterable, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(iterable, batch_size))


# data injestion
# batch injestion
def ingest_in_batches(file, file_name, batch_size: int = 100):
    """injests vectors in batches to reduce memory usage"""
    for batch in chunks(file, batch_size):
        # metadata for the vector database
        metadata = [
            {
                'source': file_name,
                'texts': batch
            }
        ]

        # id to reference batch
        batch_id = str(uuid.uuid4())
        embeds = np.array(create_embeddings(batch))
        vector_db_client.upsert(
            vectors=zip(batch_id, embeds, metadata),
            namespace=namespace)


# parallel ingesting
def ingest_in_parallel(file, file_name, pool_threads: int = 30, batch_size: int = 100):
    """Submits upsert requests asynchronously for each chunk"""

    with vector_db_client.Index(index_name, pool_threads = pool_threads) as index:
        async_results = []
        for batch in chunks(file, batch_size):
            # metadata for the vector database
            metadata = [
                {
                    'source': file_name,
                    'texts': batch
                }
            ]

            # id to reference batch
            batch_id = str(uuid.uuid4())

            # embeddings
            embeds = np.array(create_embeddings(batch))

            # collect results
            async_results.append(
                index.upsert(
                    vectors=zip(batch_id, embeds, metadata),
                    namespace=namespace,
                    async_req=True,
                )
            )
        result = [results.get() for results in async_results]
    return result

