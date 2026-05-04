"""Defines API endpoints"""
import asyncio
from fastapi import HTTPException, APIRouter

# custom modules
from ...core.utils.startup import connection_success
from ..middleware.schema import UserRequest, ChatResponse
from ...core.generation.rag_pipeline import RAGPipeline
from ...core.utils.helpers import timer, timeout
from ...core.utils.logger import logger


def router():
    """serves router for bot related endpoints"""
    return APIRouter()


chatbot = router()


@timeout(30)
@chatbot.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_request: UserRequest):
    """endpoint to facilitate conversations"""
    try:
        chat = RAGPipeline()
        response = await asyncio.to_thread(
            chat.generate_answer, user_request.prompt)
        logger.info("results returned successfully")
        await asyncio.sleep(1)  # Simulate delay
        return ChatResponse(
            response=response['answer'], sources=response['sources'])
    except Exception as exc:
        logger.error("unsuccessful", extra={"error": type(exc).__name__})
        raise HTTPException(
            status_code=500, detail="could not provide a response") from exc


@timer
@chatbot.get("/services/health")
async def service_health():
    """gets initialization status of core resources"""
    try:
        status = await asyncio.wait_for(
            connection_success(), timeout=10)
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
        logger.error(
            "Health check failed", extra={"error": type(exc).__name__})
        raise HTTPException(
            status_code=500, detail="Service health check failed") from exc
