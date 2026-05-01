"""API server entry point"""

"""This is a simple FastAPI application that uses the Hugging Face Inference API to generate text based on a given prompt.
The application defines an endpoint that accepts a prompt and returns the generated text."""


import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pinecone import Pinecone
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt)


from ..core.utils.startup import make_connections
from ..routes.chatbot import router

# making resources available upon startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """initializes resources on startup"""
    # Initialize resources here
    await make_connections()
    yield app
    # Clean up resources here


# api client
@retry(wait=wait_random_exponential(10, 40), stop=stop_after_attempt(5))
def create_app():
    """creates the client application and includes the routes"""
    app = FastAPI(lifespan=lifespan)
    # include API endpoints
    app.include_router(router(), prefix="/v1")
    return app


# client application
server = create_app()

if __name__ == '__main__':
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
