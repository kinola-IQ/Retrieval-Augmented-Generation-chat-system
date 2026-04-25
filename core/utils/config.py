"""Module for Centralized configuration"""

import os
from dotenv import load_dotenv

# creating access to env variables
load_dotenv()

# Hugging face config
def huggingface_config():
    """serves configs for huggingface"""
    api_key = os.environ["HUGGINGFACE_API_KEY"]
    model_name: str  = "facebook/bart-large-cnn"
    task: str= 'summarization'

    return api_key, model_name, task