"""Module for Centralized configuration"""

import os
from dotenv import load_dotenv

# creating access to env variables
load_dotenv()


# Hugging face config
def huggingface_config():
    """serves configs for huggingface"""
    api_key = os.environ["HUGGINGFACE_API_KEY"]
    model_name = os.environ['HUGGINGFACE_MODEL_NAME']
    task = os.environ['HUGGINGFACE_TASK']
    embedding_model = os.environ['HUGGINGFACE_EMBEDDING_MODEL']
    return {
        'api key': api_key,
        'model': model_name,
        'task': task,
        'embedding model': embedding_model}


# pinecone config
def pinecone_config():
    """serves configs for pinecone vector database"""
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]
    index_name = os.environ["PINECONE_INDEX_NAME"]
    namespace = os.environ["PINECONE_NAMESPACE"]
    index = os.environ['PINECONE_INDEX']
    return {
        'api key': api_key,
        'environment': environment,
        'index': index_name,
        'namespace': namespace,
        'index': index}
