"""Module for Centralized configuration"""

import os
from dotenv import load_dotenv

# creating access to env variables
load_dotenv()

hugging_face_api_key = os.environ["HUGGINGFACE_API_KEY"]