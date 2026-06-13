"""module to handle the backend cold start"""
import os
from dotenv import load_dotenv
import socket
import threading
import streamlit as st
import uvicorn

from core.utils.logger import logger

load_dotenv()
# api configurations
host = os.environ.get("HOST","0.0.0.0")
port = os.environ.get("PORT", 8000)


# backend startup
def port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, int(port))) == 0


def run_backend():
    logger.info(
        "streamlit_backend_start"
    )
    uvicorn.run(
        "api.server:server",
        host=host,
        port=int(port),
        reload=False
    )


@st.cache_resource
def start_backend():
     # we need the server up and running
     # for requests to go through to the backend
    if not port_in_use(host, port):
        threading.Thread(
            target=run_backend,
            daemon=True
        ).start()
        return True
    