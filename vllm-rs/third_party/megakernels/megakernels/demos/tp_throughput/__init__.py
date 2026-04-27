# This module contains the throughput-optimized tensor parallel implementation
# The C++ scheduler is now in a separate submodule to avoid automatic loading
#
# To use the C++ scheduler:
#   from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
#
# Or to maintain backward compatibility in your code:
#   from megakernels.demos.tp_throughput import cpp_scheduler
#   scheduler_cpp = cpp_scheduler.scheduler_cpp
