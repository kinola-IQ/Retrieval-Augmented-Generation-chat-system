"""Defines API endpoints"""
import asyncio
from pathlib import Path
from fastapi import HTTPException, APIRouter, BackgroundTasks


# custom modules
from core.utils import startup
from core.utils.helpers import timer, timeout, export_to_csv
from core.utils.logger import logger
from core.utils.config import benchmark_const
from core.utils.constants import home_path
from ..middleware.schema import UserRequest, ChatResponse


chatbot = APIRouter()


@timeout(30)
@chatbot.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
        user_request: UserRequest, background_tasks: BackgroundTasks):
    """endpoint to facilitate conversations"""
    try:
        # we create an instance of the RAGPipeline
        # then run the pipeline to generate an answer
        chat = startup.BOT
        response = await chat.generate_answer(user_request.prompt)

        # export interaction for evaluation
        async def store_interaction(prompt, answer, source_text):
            try:
                data = [prompt, answer, source_text]
                await export_to_csv(data)
                logger.info("background task completed: data exported")
            except Exception:
                logger.exception("failed to export interaction")

        source_text = ""
        if response.get("sources"):
            for references in response["sources"]:
                source_text += references.get("text", "")
        # activated only when running locally
        if (home_path() / "tests" / "experimental.py").exists():
            background_tasks.add_task(
                store_interaction,
                user_request.prompt,
                response.get("answer", ""),
                source_text,
            )

        logger.info("results returned successfully")
        await asyncio.sleep(1)  # Simulate delay
        return ChatResponse(
            response=response['answer'], sources=response['sources'])
    except Exception as exc:
        logger.exception("could not generate an answer", exc_info=True)
        raise HTTPException(
            status_code=500, detail="could not provide a response") from exc


@timer
@chatbot.get("/services/health")
async def service_health():
    """gets initialization status of core resources"""
    try:
        # we wait for the connection to be successful
        # this is a blocking operation
        # so we use asyncio.wait_for to timeout after 10 seconds
        status = await asyncio.wait_for(
            startup.connection_success(), timeout=10)
        if status is False:
            logger.warning(
                "Health check failed: external services not initialized")
            raise HTTPException(status_code=500, detail="Service not ready")

        return {
            "status": "healthy",
            "service": "running",
            "services_initialized": status
        }
    except asyncio.TimeoutError as exc:
        logger.error("Health check failed: timeout while checking services")
        raise HTTPException(
            status_code=408, detail="Service health check timed out") from exc

    except Exception as exc:
        logger.exception(
            "Health check failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Service health check failed") from exc
