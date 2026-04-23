"""Handles request validation"""
from pydantic import BaseModel

class UserRequest(BaseModel):
    prompt: str 