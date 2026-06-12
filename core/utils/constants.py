"""holds the immutable values needed for critical bottlenecks"""
from pathlib import Path


def home_path():
    """Returns the home directry for the repository"""
    return Path(__file__).resolve().parents[2]
