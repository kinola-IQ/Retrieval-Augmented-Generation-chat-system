"""Defines API endpoints"""
from fastapi import HTTPException, APIRouter
import request

# custom modules
from ..core.utils.startup import get_resources
from ..server import server

def router():
    """serves router for bot related endpoints"""
    return APIRouter()