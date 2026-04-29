"""Module for Misc utilities"""
import time
import threading
from functools import wraps

# custom imports
from .logger import logger


# for timeout decorator
def raise_timeout():
    """Helper function to raise TimeoutError after a delay."""
    # This flag will be set if timeout occurs
    raise TimeoutError


# decorators

# for graceful shutdown of long-running functions
def timeout(seconds: int):
    """Decorator to raise TimeoutError \
        if function runs longer than `seconds`."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            shutdown_timer = threading.Timer(seconds, raise_timeout)
            shutdown_timer.start()
            try:
                result = func(*args, **kwargs)
                logger.info(
                    "Function %s completed within timeout.",
                    func.__name__)
                return result
            except TimeoutError:
                logger.info(
                    "Function %s timed out after %s seconds.",
                    func.__name__, seconds)
            finally:
                shutdown_timer.cancel()
        return wrapper
    return decorator


# to keep track of time to execute
def timer(func):
    """times the execution of a function and logs it."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(
            "Function %s executed in %s seconds.",
            func.__name__, elapsed_time
        )
        return result
    return wrapper


# to affirm return type of a function
def returns(return_type):
    """Decorator to check if a function returns the expected type."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, return_type):
                logger.error(
                    "Function %s returned type %s, expected %s.",
                    func.__name__, type(result).__name__, return_type.__name__
                )
                raise TypeError(
                    f"Expected return type {return_type.__name__}, "
                    f"but got {type(result).__name__}."
                )
            logger.info(
                "Function %s returned expected type %s.",
                func.__name__, return_type.__name__
            )
            return result
        return wrapper
    return decorator
