"""
C++ Scheduler Module - Load this explicitly when you need the C++ scheduling functionality.

Usage:
    from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
    
    # Or to check if it's available:
    from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp, is_available
    
    if is_available():
        # Use C++ scheduler
        result = scheduler_cpp.create_instruction_tensor(...)
    else:
        # Fall back to Python implementation
        pass
"""

import os
import warnings
from pathlib import Path

import pybind11
from torch.utils import cpp_extension

# Control via environment variable
USE_CPP_SCHEDULER = os.environ.get("USE_CPP_SCHEDULER", "1") == "1"

scheduler_cpp = None
_load_attempted = False
_load_error = None


def _load_cpp_extension():
    """Load the C++ scheduler extension."""
    global scheduler_cpp, _load_attempted, _load_error
    
    if _load_attempted:
        return scheduler_cpp
    
    _load_attempted = True
    
    if not USE_CPP_SCHEDULER:
        _load_error = "C++ scheduler disabled via USE_CPP_SCHEDULER=0"
        return None
    
    try:
        # Get the path to the cpp_src directory
        cpp_src_dir = Path(__file__).parent / "cpp_src"
        
        print(f"Loading C++ scheduler extension from {cpp_src_dir}")
        
        # Build and load the extension
        scheduler_cpp = cpp_extension.load(
            name="scheduler_cpp",
            sources=[
                str(cpp_src_dir / "bindings.cpp"),
                str(cpp_src_dir / "scheduling.cpp"),
            ],
            extra_cflags=["-std=c++17", "-O3"],
            extra_ldflags=[],
            extra_include_paths=[pybind11.get_include()],
            verbose=False,  # Set to True for debugging
        )
        
        print("✅ C++ scheduler extension loaded successfully")
        return scheduler_cpp
        
    except Exception as e:
        _load_error = str(e)
        warnings.warn(f"Failed to load C++ scheduler extension: {e}")
        scheduler_cpp = None
        return None


def is_available():
    """Check if the C++ scheduler is available."""
    if scheduler_cpp is None and not _load_attempted:
        _load_cpp_extension()
    return scheduler_cpp is not None


def get_scheduler():
    """Get the C++ scheduler module, loading it if necessary."""
    if scheduler_cpp is None and not _load_attempted:
        _load_cpp_extension()
    
    if scheduler_cpp is None:
        raise RuntimeError(f"C++ scheduler not available. Error: {_load_error}")
    
    return scheduler_cpp


# Convenience: automatically load on module import
# Users can still check is_available() before using
scheduler_cpp = _load_cpp_extension()

__all__ = ["scheduler_cpp", "is_available", "get_scheduler"]