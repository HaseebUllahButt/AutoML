"""
Comprehensive error handling utilities with timeout, retry, and validation decorators
Provides context managers, custom exceptions, and recovery strategies
"""

import functools
import time
import signal
import logging
import traceback
from typing import Callable, Any, Optional, Type, Union, Tuple
from contextlib import contextmanager
import inspect


# ==================== CUSTOM EXCEPTIONS ====================

class TimeoutError(Exception):
    """Raised when operation times out"""
    pass


class AutoMLException(Exception):
    """Base exception for all AutoML errors"""
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert exception to dictionary for logging"""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'context': self.context
        }


class DataValidationError(AutoMLException):
    """Raised when data validation fails"""
    pass


class IngestException(DataValidationError):
    """Raised when data ingestion fails"""
    pass


class ProfilingException(AutoMLException):
    """Raised when data profiling fails"""
    pass


class PreprocessingException(AutoMLException):
    """Raised when preprocessing fails"""
    pass


class TrainingException(AutoMLException):
    """Raised when model training fails"""
    pass


class ReportException(AutoMLException):
    """Raised when report generation fails"""
    pass


class ConfigurationError(AutoMLException):
    """Raised when configuration is invalid"""
    pass


# ==================== ERROR CONTEXT MANAGER ====================

@contextmanager
def ErrorContext(operation_name: str, logger: Optional[logging.Logger] = None, 
                 critical: bool = False):
    """
    Context manager for comprehensive error tracking
    
    Usage:
        with ErrorContext("data loading"):
            df = load_data("file.csv")
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"✓ {operation_name} completed in {duration:.2f}s")
    
    except Exception as e:
        duration = time.time() - start_time
        error_info = {
            'operation': operation_name,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'duration': f"{duration:.2f}s",
            'traceback': traceback.format_exc()
        }
        
        if critical:
            logger.critical(f"✗ CRITICAL {operation_name} failed: {e}", extra=error_info)
        else:
            logger.error(f"✗ {operation_name} failed: {e}", extra=error_info)
        
        raise


# ==================== VALIDATION DECORATOR ====================

def validate_inputs(**type_specs):
    """
    Decorator to validate input parameter types
    
    Usage:
        @validate_inputs(X=(pd.DataFrame, np.ndarray), y=(pd.Series, np.ndarray))
        def process_data(X, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, expected_types in type_specs.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None:
                        if not isinstance(value, expected_types):
                            raise DataValidationError(
                                f"Parameter '{param_name}' must be {expected_types}, "
                                f"got {type(value).__name__}",
                                {'function': func.__name__, 'parameter': param_name}
                            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ==================== RANGE VALIDATION DECORATOR ====================

def validate_ranges(**range_specs):
    """
    Decorator to validate parameter ranges
    
    Usage:
        @validate_ranges(max_iterations=(1, 1000), learning_rate=(0, 1))
        def train_model(max_iterations, learning_rate):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, (min_val, max_val) in range_specs.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and (value < min_val or value > max_val):
                        raise DataValidationError(
                            f"Parameter '{param_name}' must be between {min_val} and {max_val}, "
                            f"got {value}",
                            {'function': func.__name__, 'parameter': param_name}
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ==================== TIMEOUT DECORATOR ====================

def timeout(seconds: int):
    """
    Decorator to add timeout to a function (Unix-like systems only)
    
    Usage:
        @timeout(300)
        def long_running_function():
            ...
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Define timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Function '{func.__name__}' timed out after {seconds} seconds",
                    {'function': func.__name__, 'timeout_seconds': seconds}
                )
            
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


# ==================== RETRY DECORATOR WITH BACKOFF ====================

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 1.0,
          exceptions: Tuple[Type[Exception], ...] = (Exception,),
          logger: Optional[logging.Logger] = None):
    """
    Decorator to retry a function on failure with exponential backoff
    
    Usage:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def flaky_function():
            ...
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier (e.g., 2.0 for exponential)
        exceptions: Tuple of exception types to catch
        logger: Logger instance for tracking retries
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    last_exception = e
                    
                    if attempts < max_attempts:
                        logger.warning(
                            f"Attempt {attempts}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# ==================== SAFE EXECUTION ====================

def safe_execute(func: Callable, *args, default: Any = None, 
                 logger: Optional[logging.Logger] = None,
                 **kwargs) -> Any:
    """
    Safely execute a function and return default on error
    
    Usage:
        result = safe_execute(risky_function, arg1, arg2, default=0)
    
    Args:
        func: Function to execute
        default: Default value to return on error
        logger: Logger instance for error tracking
        
    Returns:
        Function result or default value on error
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return default


# ==================== ERROR COLLECTOR ====================

class ErrorCollector:
    """
    Collects errors and warnings during execution without stopping
    Useful for batch processing where you want to continue despite some failures
    """
    
    def __init__(self, max_errors: Optional[int] = None):
        self.errors = []
        self.warnings = []
        self.infos = []
        self.max_errors = max_errors
    
    def add_error(self, error: str, context: Optional[dict] = None):
        """Add an error"""
        if self.max_errors is None or len(self.errors) < self.max_errors:
            self.errors.append({'message': error, 'context': context or {}})
    
    def add_warning(self, warning: str, context: Optional[dict] = None):
        """Add a warning"""
        self.warnings.append({'message': warning, 'context': context or {}})
    
    def add_info(self, info: str):
        """Add informational message"""
        self.infos.append(info)
    
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if number of errors exceeds max_errors"""
        return self.max_errors is not None and len(self.errors) >= self.max_errors
    
    def error_count(self) -> int:
        """Get number of errors"""
        return len(self.errors)
    
    def warning_count(self) -> int:
        """Get number of warnings"""
        return len(self.warnings)
    
    def get_summary(self) -> str:
        """Get a formatted summary of all errors and warnings"""
        summary = []
        
        if self.errors:
            summary.append(f"❌ Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                msg = error['message'] if isinstance(error, dict) else error
                summary.append(f"  - {msg}")
            if len(self.errors) > 10:
                summary.append(f"  ... and {len(self.errors) - 10} more errors")
        
        if self.warnings:
            summary.append(f"⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                msg = warning['message'] if isinstance(warning, dict) else warning
                summary.append(f"  - {msg}")
            if len(self.warnings) > 10:
                summary.append(f"  ... and {len(self.warnings) - 10} more warnings")
        
        if self.infos:
            summary.append(f"ℹ️ Info ({len(self.infos)}):")
            for info in self.infos[:5]:  # Show first 5
                summary.append(f"  - {info}")
            if len(self.infos) > 5:
                summary.append(f"  ... and {len(self.infos) - 5} more info messages")
        
        return '\n'.join(summary) if summary else "✓ No errors or warnings"
    
    def to_dict(self) -> dict:
        """Export all collected messages as dictionary"""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'infos': self.infos,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.infos)
        }


# ==================== INPUT VALIDATION HELPERS ====================

class InputValidator:
    """Helper class for validating various input types"""
    
    @staticmethod
    def validate_not_none(*args, **kwargs) -> bool:
        """Check that all arguments are not None"""
        for arg in args:
            if arg is None:
                return False
        for value in kwargs.values():
            if value is None:
                return False
        return True
    
    @staticmethod
    def validate_not_empty(obj, obj_name: str = "object") -> bool:
        """Check that object is not empty"""
        try:
            return len(obj) > 0
        except:
            return obj is not None and obj != ""
    
    @staticmethod
    def validate_shape(arr, expected_shape: Tuple[int, ...], strict: bool = True) -> bool:
        """
        Validate array shape
        
        Args:
            arr: Array to validate
            expected_shape: Expected shape (use -1 for any dimension)
            strict: If True, all dimensions must match. If False, at least match specified dims
        """
        if not hasattr(arr, 'shape'):
            return False
        
        if strict:
            if len(arr.shape) != len(expected_shape):
                return False
            for actual, expected in zip(arr.shape, expected_shape):
                if expected != -1 and actual != expected:
                    return False
        else:
            for actual, expected in zip(arr.shape, expected_shape):
                if expected != -1 and actual != expected:
                    return False
        
        return True
