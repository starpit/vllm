// This crate has no runtime code — it exists solely to compile CUDA kernels
// via build.rs. The compiled .a files land in the shared cudaforge cache and
// are linked by vllm-cuda.
