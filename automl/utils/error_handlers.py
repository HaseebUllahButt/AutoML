"""
Error handling utilities with timeout and retry decorators
"""

import functools
import time
import signal
from typing import Callable, Any


class TimeoutError(Exception):
    """Raised when operation times out"""
    pass


def timeout(seconds: int):
    """
    Decorator to add timeout to a function
    
    Usage:
        @timeout(300)
        def long_running_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Define timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
            
            # Set up alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry a function on failure
    
    Usage:
        @retry(max_attempts=3, delay=2.0)
        def flaky_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    last_exception = e
                    if attempts < max_attempts:
                        time.sleep(delay)
            
            # If all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default=None, **kwargs) -> Any:
    """
    Safely execute a function and return default on error
    
    Usage:
        result = safe_execute(risky_function, arg1, arg2, default=0)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return default


class ErrorCollector:
    """Collects errors during execution without stopping"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, error: str):
        """Add an error"""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning"""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0
    
    def get_summary(self) -> str:
        """Get a summary of all errors and warnings"""
        summary = []
        
        if self.errors:
            summary.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                summary.append(f"  - {error}")
        
        if self.warnings:
            summary.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                summary.append(f"  - {warning}")
        
        return '\n'.join(summary) if summary else "No errors or warnings"
