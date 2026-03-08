// SPDX-License-Identifier: Apache-2.0
//! cuBLAS handle with cached GEMM plans for zero-overhead repeated calls.
//!
//! All GEMMs go through **cublasLt** with heuristic algorithm selection.
//! Plans (descriptors + algorithm) are cached by `(M, K, N, dtype, has_bias)`
//! so that the first call creates the plan and subsequent calls with the same
//! shapes reuse it — eliminating ~20 API calls per GEMM.
//!
//! The workspace is bound to the handle via `cublasSetWorkspace()` at init,
//! so cuBLAS never lazily allocates on the default stream — this is what
//! makes CUDA graph capture work.

use std::collections::HashMap;

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::{Result, bail};
use cudarc::cublas::sys::{self, cublasComputeType_t, cublasHandle_t, cublasOperation_t};
use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys::CUstream;

/// cuBLAS workspace size (4 MB — matches Python vLLM).
const CUBLAS_WORKSPACE_SIZE: usize = 4 * 1024 * 1024;

/// Cache key for a GEMM plan.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct PlanKey {
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    has_bias: bool,
}

/// A cached cublasLt GEMM plan — holds pre-created descriptors and the best algorithm.
struct GemmPlan {
    matmul_desc: lt::cublasLtMatmulDesc_t,
    layout_a: lt::cublasLtMatrixLayout_t,
    layout_b: lt::cublasLtMatrixLayout_t,
    layout_c: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
}

impl Drop for GemmPlan {
    fn drop(&mut self) {
        unsafe {
            lt::cublasLtMatrixLayoutDestroy(self.layout_a);
            lt::cublasLtMatrixLayoutDestroy(self.layout_b);
            lt::cublasLtMatrixLayoutDestroy(self.layout_c);
            lt::cublasLtMatmulDescDestroy(self.matmul_desc);
        }
    }
}

/// cuBLAS + cublasLt handles bound to a non-default stream with pre-allocated workspace.
pub struct CublasHandle {
    handle: cublasHandle_t,
    lt_handle: lt::cublasLtHandle_t,
    stream: CUstream,
    workspace: *mut u8,
    /// Cached GEMM plans keyed by (M, K, N, dtype, has_bias).
    plans: HashMap<PlanKey, GemmPlan>,
}

// Safety: cuBLAS handle is thread-safe when each thread uses its own handle
// or synchronizes externally. We use one handle per GpuDevice.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Create a new cuBLAS handle bound to `stream` with a pre-allocated workspace.
    ///
    /// # Safety
    /// `stream` must be a valid non-default CUDA stream.
    pub unsafe fn new(stream: CUstream) -> Result<Self> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        check(sys::cublasCreate_v2(&mut handle))?;

        // Bind to our non-default stream.
        check(sys::cublasSetStream_v2(handle, stream as _))?;

        // Enable TF32 tensor cores (2x throughput for F32 on Ampere+).
        check(sys::cublasSetMathMode(
            handle,
            sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        ))?;

        // Pre-allocate workspace (fixes CUDA graph capture).
        let workspace = driver::mem_alloc(CUBLAS_WORKSPACE_SIZE)?;
        check(sys::cublasSetWorkspace_v2(
            handle,
            workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
        ))?;

        // Create cublasLt handle.
        let mut lt_handle: lt::cublasLtHandle_t = std::ptr::null_mut();
        check_lt(lt::cublasLtCreate(&mut lt_handle))?;

        Ok(Self {
            handle,
            lt_handle,
            stream,
            workspace,
            plans: HashMap::new(),
        })
    }

    /// Ensure a cached GEMM plan exists for the given shapes, creating it if needed.
    unsafe fn ensure_plan(&mut self, m: usize, k: usize, n: usize, dtype: DType, has_bias: bool) {
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias,
        };
        if self.plans.contains_key(&key) {
            return;
        }

        let (_compute_type, data_type) = gemm_types(dtype);
        let lt_data_type = cublas_to_lt_dtype(data_type);
        let lt_compute_type = cublas_to_lt_compute(_compute_type);

        // Create matmul descriptor.
        let mut matmul_desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulDescCreate(
            &mut matmul_desc,
            lt_compute_type,
            lt::cudaDataType_t::CUDA_R_32F, // scale type always F32
        ))
        .expect("cublasLtMatmulDescCreate failed");

        // Set transpose operations: C^T = B @ A^T in col-major → C = A @ B^T in row-major.
        let transa = cublasOperation_t::CUBLAS_OP_T as i32;
        let transb = cublasOperation_t::CUBLAS_OP_N as i32;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("set TRANSA failed");
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("set TRANSB failed");

        // Set bias epilogue if needed.
        if has_bias {
            let epilogue = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            check_lt(lt::cublasLtMatmulDescSetAttribute(
                matmul_desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue as *const _ as *const _,
                std::mem::size_of_val(&epilogue),
            ))
            .expect("set EPILOGUE failed");
        }

        // Create matrix layouts (column-major convention: C^T = B @ A^T).
        // A in cuBLAS = B (weight) [N, K] col-major → rows=K, cols=N, ld=K
        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_a,
            lt_data_type,
            k as u64,
            n as u64,
            k as i64,
        ))
        .expect("layout A");

        // B in cuBLAS = A (activation) [M, K] col-major → rows=K, cols=M, ld=K
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_b,
            lt_data_type,
            k as u64,
            m as u64,
            k as i64,
        ))
        .expect("layout B");

        // C/D = output [M, N] col-major → rows=N, cols=M, ld=N
        let mut layout_c: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_c,
            lt_data_type,
            n as u64,
            m as u64,
            n as i64,
        ))
        .expect("layout C");

        // Get heuristic for best algorithm.
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulPreferenceCreate(&mut pref)).expect("pref create");
        let ws_size = CUBLAS_WORKSPACE_SIZE;
        check_lt(lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const _ as *const _,
            std::mem::size_of::<usize>(),
        ))
        .expect("set pref workspace");

        let mut heuristic = std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>();
        let mut algo_count: i32 = 0;
        check_lt(lt::cublasLtMatmulAlgoGetHeuristic(
            self.lt_handle,
            matmul_desc,
            layout_a,
            layout_b,
            layout_c,
            layout_c, // D layout = C layout
            pref,
            1,
            &mut heuristic,
            &mut algo_count,
        ))
        .expect("cublasLtMatmulAlgoGetHeuristic failed");
        assert!(algo_count > 0, "no cublasLt algorithm found");

        lt::cublasLtMatmulPreferenceDestroy(pref);

        let plan = GemmPlan {
            matmul_desc,
            layout_a,
            layout_b,
            layout_c,
            algo: heuristic.algo,
        };

        self.plans.insert(key, plan);
    }

    /// GEMM: out = A @ B^T
    ///
    /// - `a`: `[M, K]` row-major
    /// - `b`: `[N, K]` row-major (weight stored as `[out_features, in_features]`)
    /// - Returns: `[M, N]` allocated from `arena`
    ///
    /// Uses cublasLt with cached plans for optimal algorithm selection,
    /// especially split-K for decode-regime skinny GEMMs (M=1-8).
    ///
    /// # Safety
    /// `a` and `b` must be valid GPU tensors with compatible dtypes and shapes.
    pub unsafe fn gemm(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> GpuTensor {
        self.gemm_owned(a, b, alloc).into_gpu_tensor()
    }

    /// GEMM with fused bias add: out = A @ B^T + bias
    ///
    /// Uses cublasLt with `CUBLASLT_EPILOGUE_BIAS` to fuse the bias add into
    /// the GEMM kernel — zero extra kernel launches, zero extra memory traffic.
    ///
    /// - `a`: `[M, K]` row-major (activations)
    /// - `b`: `[N, K]` row-major (weight, transposed internally)
    /// - `bias`: `[N]` (broadcast along M dimension)
    /// - Returns: `[M, N]` allocated from `arena`
    ///
    /// # Safety
    /// All tensors must be valid GPU memory with compatible dtypes.
    pub unsafe fn gemm_bias(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        bias: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> GpuTensor {
        self.gemm_bias_owned(a, b, bias, alloc).into_gpu_tensor()
    }

    /// GEMM returning `OwnedTensor` allocated from caching allocator.
    ///
    /// Same semantics as `gemm()` but memory is freed on drop.
    pub unsafe fn gemm_owned(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");

        let m = a.dim(0);
        let k = a.dim(1);
        let n = b.dim(0);
        let dtype = a.dtype();

        let out = alloc.alloc_tensor(&[m, n], dtype);

        self.ensure_plan(m, k, n, dtype, false);
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias: false,
        };
        let plan = &self.plans[&key];

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        check_lt(lt::cublasLtMatmul(
            self.lt_handle,
            plan.matmul_desc,
            &alpha as *const f32 as *const _,
            b.as_ptr::<u8>() as *const _,
            plan.layout_a,
            a.as_ptr::<u8>() as *const _,
            plan.layout_b,
            &beta as *const f32 as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            &plan.algo,
            self.workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
            self.stream as _,
        ))
        .expect("cublasLtMatmul failed");

        out
    }

    /// GEMM with fused bias add, returning `OwnedTensor`.
    pub unsafe fn gemm_bias_owned(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        bias: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(bias.ndim(), 1);
        debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
        debug_assert_eq!(bias.dim(0), b.dim(0), "bias size must match N");

        let m = a.dim(0);
        let k = a.dim(1);
        let n = b.dim(0);
        let dtype = a.dtype();

        let out = alloc.alloc_tensor(&[m, n], dtype);

        self.ensure_plan(m, k, n, dtype, true);
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias: true,
        };
        let plan = &self.plans[&key];

        let bias_ptr = bias.raw_ptr() as *const std::ffi::c_void;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            plan.matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr as *const _ as *const _,
            std::mem::size_of::<*const std::ffi::c_void>(),
        ))
        .expect("set BIAS_POINTER failed");

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        check_lt(lt::cublasLtMatmul(
            self.lt_handle,
            plan.matmul_desc,
            &alpha as *const f32 as *const _,
            b.as_ptr::<u8>() as *const _,
            plan.layout_a,
            a.as_ptr::<u8>() as *const _,
            plan.layout_b,
            &beta as *const f32 as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            &plan.algo,
            self.workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
            self.stream as _,
        ))
        .expect("cublasLtMatmul failed");

        out
    }

    /// Benchmark all cached GEMM plans: for each shape, try the top N
    /// algorithms from the heuristic and keep the fastest.
    ///
    /// Call this during warmup after the first forward pass has populated
    /// the plan cache with all model GEMM shapes.
    ///
    /// # Safety
    /// Requires active CUDA context. Arena must have enough space for the
    /// largest GEMM output.
    pub unsafe fn benchmark_plans(&mut self) {
        use crate::driver;

        let keys: Vec<PlanKey> = self.plans.keys().copied().collect();
        if keys.is_empty() {
            return;
        }

        tracing::info!(
            "Benchmarking {} cublasLt GEMM plans ({} warmup + {} timed iters)...",
            keys.len(),
            3,
            10
        );

        let (event_start, event_end) = match (driver::event_create(), driver::event_create()) {
            (Ok(s), Ok(e)) => (s, e),
            _ => {
                tracing::warn!("Failed to create CUDA events for benchmarking, skipping");
                return;
            }
        };

        for key in &keys {
            // Allocate scratch buffers for benchmarking.
            let a_bytes = key.m * key.k * key.dtype.size_bytes();
            let b_bytes = key.n * key.k * key.dtype.size_bytes();
            let c_bytes = key.m * key.n * key.dtype.size_bytes();
            let total = a_bytes + b_bytes + c_bytes;

            // Use temp allocations (not arena, which may be sized for the model).
            let Ok(buf) = driver::mem_alloc(total) else {
                continue;
            };
            let a_ptr = buf;
            let b_ptr = unsafe { buf.add(a_bytes) };
            let c_ptr = unsafe { buf.add(a_bytes + b_bytes) };
            let _ = driver::memset_d8(buf, 0, total, self.stream);

            // Get the plan's descriptors.
            let plan = &self.plans[key];
            let matmul_desc = plan.matmul_desc;
            let layout_a = plan.layout_a;
            let layout_b = plan.layout_b;
            let layout_c = plan.layout_c;

            // Get top N algorithms.
            let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
            if check_lt(lt::cublasLtMatmulPreferenceCreate(&mut pref)).is_err() {
                let _ = driver::mem_free(buf);
                continue;
            }
            let ws_size = CUBLAS_WORKSPACE_SIZE;
            let _ = check_lt(lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const _,
                std::mem::size_of::<usize>(),
            ));

            const MAX_ALGOS: usize = 8;
            let mut heuristics =
                [std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>(); MAX_ALGOS];
            let mut algo_count: i32 = 0;
            let _ = lt::cublasLtMatmulAlgoGetHeuristic(
                self.lt_handle,
                matmul_desc,
                layout_a,
                layout_b,
                layout_c,
                layout_c,
                pref,
                MAX_ALGOS as i32,
                heuristics.as_mut_ptr(),
                &mut algo_count,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);

            if algo_count <= 1 {
                // Only one algorithm available, nothing to benchmark.
                let _ = driver::mem_free(buf);
                continue;
            }

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let mut best_time = f32::MAX;
            let mut best_algo_idx = 0usize;
            let mut heuristic_time = f32::MAX; // time for algo 0 (heuristic pick)

            for (ai, heuristic) in heuristics.iter().enumerate().take(algo_count as usize) {
                let algo = &heuristic.algo;

                // Warmup.
                for _ in 0..3 {
                    let _ = lt::cublasLtMatmul(
                        self.lt_handle,
                        matmul_desc,
                        &alpha as *const f32 as *const _,
                        b_ptr as *const _,
                        layout_a,
                        a_ptr as *const _,
                        layout_b,
                        &beta as *const f32 as *const _,
                        c_ptr as *mut _,
                        layout_c,
                        c_ptr as *mut _,
                        layout_c,
                        algo,
                        self.workspace as *mut _,
                        CUBLAS_WORKSPACE_SIZE,
                        self.stream as _,
                    );
                }

                // Timed iterations.
                let _ = driver::event_record(event_start, self.stream);
                for _ in 0..10 {
                    let _ = lt::cublasLtMatmul(
                        self.lt_handle,
                        matmul_desc,
                        &alpha as *const f32 as *const _,
                        b_ptr as *const _,
                        layout_a,
                        a_ptr as *const _,
                        layout_b,
                        &beta as *const f32 as *const _,
                        c_ptr as *mut _,
                        layout_c,
                        c_ptr as *mut _,
                        layout_c,
                        algo,
                        self.workspace as *mut _,
                        CUBLAS_WORKSPACE_SIZE,
                        self.stream as _,
                    );
                }
                let _ = driver::event_record(event_end, self.stream);
                let _ = driver::stream_synchronize(self.stream);

                if let Ok(elapsed) = driver::event_elapsed(event_start, event_end) {
                    let avg = elapsed / 10.0;
                    if ai == 0 {
                        heuristic_time = avg;
                    }
                    if avg < best_time {
                        best_time = avg;
                        best_algo_idx = ai;
                    }
                }
            }

            // Update the plan with the best algorithm.
            if best_algo_idx != 0 {
                let plan = self.plans.get_mut(key).unwrap();
                plan.algo = heuristics[best_algo_idx].algo;
                let speedup = (1.0 - best_time / heuristic_time) * 100.0;
                tracing::info!(
                    "  GEMM M={} K={} N={}: algo #{} is {:.1}% faster ({:.3}ms vs {:.3}ms)",
                    key.m,
                    key.k,
                    key.n,
                    best_algo_idx,
                    speedup,
                    best_time,
                    heuristic_time,
                );
            }

            let _ = driver::mem_free(buf);
        }

        let _ = driver::event_destroy(event_start);
        let _ = driver::event_destroy(event_end);

        tracing::info!("cublasLt plan benchmarking complete");
    }

    /// Raw cuBLAS handle (for advanced usage).
    pub fn raw_handle(&self) -> cublasHandle_t {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // Drop all cached plans first (they reference the lt_handle indirectly).
        self.plans.clear();
        unsafe {
            let _ = lt::cublasLtDestroy(self.lt_handle);
            let _ = sys::cublasDestroy_v2(self.handle);
            let _ = driver::mem_free(self.workspace);
        }
    }
}

/// Map our DType to cuBLAS compute and data types.
fn gemm_types(dtype: DType) -> (cublasComputeType_t, sys::cudaDataType_t) {
    match dtype {
        DType::F16 => (
            cublasComputeType_t::CUBLAS_COMPUTE_16F,
            sys::cudaDataType_t::CUDA_R_16F,
        ),
        DType::BF16 => (
            // BF16 uses F32 accumulation for numerical stability.
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_16BF,
        ),
        DType::F32 => (
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            sys::cudaDataType_t::CUDA_R_32F,
        ),
        _ => panic!("unsupported dtype for GEMM: {dtype}"),
    }
}

fn check(status: sys::cublasStatus_t) -> Result<()> {
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("cuBLAS error: {:?}", status);
    }
    Ok(())
}

fn check_lt(status: lt::cublasStatus_t) -> Result<()> {
    if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("cublasLt error: {:?}", status);
    }
    Ok(())
}

/// Convert cuBLAS data type to cublasLt data type (same enum values, different Rust types).
fn cublas_to_lt_dtype(dt: sys::cudaDataType_t) -> lt::cudaDataType_t {
    match dt {
        sys::cudaDataType_t::CUDA_R_16F => lt::cudaDataType_t::CUDA_R_16F,
        sys::cudaDataType_t::CUDA_R_16BF => lt::cudaDataType_t::CUDA_R_16BF,
        sys::cudaDataType_t::CUDA_R_32F => lt::cudaDataType_t::CUDA_R_32F,
        _ => panic!("unsupported dtype for cublasLt: {:?}", dt),
    }
}

/// Convert cuBLAS compute type to cublasLt compute type (same enum values, different Rust types).
fn cublas_to_lt_compute(ct: cublasComputeType_t) -> lt::cublasComputeType_t {
    match ct {
        cublasComputeType_t::CUBLAS_COMPUTE_16F => lt::cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cublasComputeType_t::CUBLAS_COMPUTE_32F => lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => {
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        }
        _ => panic!("unsupported compute type for cublasLt: {:?}", ct),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver;

    fn init_cuda() -> CUstream {
        unsafe {
            driver::init().expect("CUDA init");
            let dev = driver::device_get(0).expect("device");
            let _ctx = driver::ctx_create(dev).expect("context");
            driver::stream_create().expect("stream")
        }
    }

    #[test]
    fn test_gemm_types_f16() {
        let (compute, data) = gemm_types(DType::F16);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_16F);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_16F);
    }

    #[test]
    fn test_gemm_types_bf16() {
        let (compute, data) = gemm_types(DType::BF16);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_32F);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_16BF);
    }

    #[test]
    fn test_gemm_types_f32() {
        let (compute, data) = gemm_types(DType::F32);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_32F);
    }

    #[test]
    #[should_panic(expected = "unsupported dtype for GEMM")]
    fn test_gemm_types_u32_panics() {
        gemm_types(DType::U32);
    }

    #[test]
    #[should_panic(expected = "unsupported dtype for GEMM")]
    fn test_gemm_types_i64_panics() {
        gemm_types(DType::I64);
    }

    #[test]
    fn test_cublas_create_and_drop() {
        let stream = init_cuda();
        let handle = unsafe { CublasHandle::new(stream).unwrap() };
        assert!(!handle.raw_handle().is_null());
        drop(handle);
        unsafe { driver::stream_destroy(stream).unwrap() };
    }

    #[test]
    fn test_gemm_f32_identity() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [2, 3], B = identity-like [3, 3]
            // A @ B^T should give A when B = I (but B is [N,K] so B^T = I^T = I).
            let m = 2usize;
            let k = 3usize;
            let n = 3usize;

            // Prepare A on host: [[1, 2, 3], [4, 5, 6]]
            let host_a = driver::mem_alloc_host(m * k * 4).unwrap();
            let a_slice = std::slice::from_raw_parts_mut(host_a as *mut f32, m * k);
            a_slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

            // Prepare B (identity) on host: [[1,0,0],[0,1,0],[0,0,1]]
            let host_b = driver::mem_alloc_host(n * k * 4).unwrap();
            let b_slice = std::slice::from_raw_parts_mut(host_b as *mut f32, n * k);
            b_slice.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

            // Upload to GPU.
            let gpu_a = driver::mem_alloc(m * k * 4).unwrap();
            let gpu_b = driver::mem_alloc(n * k * 4).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, m * k * 4, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, n * k * 4, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[m, k], DType::F32);
            let b = GpuTensor::new(gpu_b, &[n, k], DType::F32);

            let c = handle.gemm(a, b, &mut arena);
            assert_eq!(c.dim(0), m);
            assert_eq!(c.dim(1), n);

            // Read back.
            let host_c = driver::mem_alloc_host(m * n * 4).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), m * n * 4, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let c_slice = std::slice::from_raw_parts(host_c as *const f32, m * n);
            // A @ I = A → [[1,2,3],[4,5,6]]
            let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            for (i, (got, exp)) in c_slice.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            // Cleanup.
            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_f32_known_values() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [[1, 2], [3, 4]] (2x2)
            // B = [[5, 6], [7, 8]] (2x2)
            // C = A @ B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
            //             = [[17, 23], [39, 53]]
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);
            let c = handle.gemm(a, b, &mut arena);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), 16, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            let expected = [17.0, 23.0, 39.0, 53.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_non_square() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [4, 8], B = [16, 8] → C = [4, 16]
            let m = 4usize;
            let k = 8usize;
            let n = 16usize;

            let gpu_a = driver::mem_alloc(m * k * 4).unwrap();
            let gpu_b = driver::mem_alloc(n * k * 4).unwrap();

            // Fill with zeros (result should be all zeros).
            driver::memset_d8(gpu_a, 0, m * k * 4, stream).unwrap();
            driver::memset_d8(gpu_b, 0, n * k * 4, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[m, k], DType::F32);
            let b = GpuTensor::new(gpu_b, &[n, k], DType::F32);
            let c = handle.gemm(a, b, &mut arena);

            assert_eq!(c.dim(0), m);
            assert_eq!(c.dim(1), n);

            // Verify output shape and that it's all zeros.
            let host_c = driver::mem_alloc_host(m * n * 4).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), m * n * 4, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, m * n);
            for (i, v) in result.iter().enumerate() {
                assert!(v.abs() < 1e-6, "expected 0 at {i}, got {v}");
            }

            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_output_from_arena() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            let mut arena = CachingAllocator::new();

            let gpu_a = driver::mem_alloc(64).unwrap();
            let gpu_b = driver::mem_alloc(64).unwrap();
            driver::memset_d8(gpu_a, 0, 64, stream).unwrap();
            driver::memset_d8(gpu_b, 0, 64, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 8], DType::F32);
            let b = GpuTensor::new(gpu_b, &[4, 8], DType::F32);

            let blocks_before = arena.total_block_count();
            let c = handle.gemm(a, b, &mut arena);
            let blocks_after = arena.total_block_count();

            // Should have allocated a new block for the output.
            assert!(blocks_after > blocks_before);
            assert_eq!(c.numel(), 8); // 2 * 4

            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_bias_f32() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [[1, 2], [3, 4]] (2x2)
            // B = [[5, 6], [7, 8]] (2x2)  (weight)
            // bias = [10, 20]
            // C = A @ B^T + bias = [[17+10, 23+20], [39+10, 53+20]] = [[27, 43], [49, 73]]
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            let host_bias = driver::mem_alloc_host(8).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
            std::slice::from_raw_parts_mut(host_bias as *mut f32, 2).copy_from_slice(&[10.0, 20.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            let gpu_bias = driver::mem_alloc(8).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_bias, host_bias, 8, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);
            let bias = GpuTensor::new(gpu_bias, &[2], DType::F32);
            let c = handle.gemm_bias(a, b, bias, &mut arena);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), 16, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            let expected = [27.0, 43.0, 49.0, 73.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "gemm_bias mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_bias).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::mem_free(gpu_bias).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_plan_caching() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream).unwrap();
            assert_eq!(handle.plans.len(), 0);

            // First call creates a plan.
            handle.ensure_plan(8, 896, 896, DType::BF16, false);
            assert_eq!(handle.plans.len(), 1);

            // Second call with same shapes reuses it.
            handle.ensure_plan(8, 896, 896, DType::BF16, false);
            assert_eq!(handle.plans.len(), 1);

            // Different shapes create a new plan.
            handle.ensure_plan(8, 896, 4864, DType::BF16, false);
            assert_eq!(handle.plans.len(), 2);

            // Same shapes but with bias create a separate plan.
            handle.ensure_plan(8, 896, 896, DType::BF16, true);
            assert_eq!(handle.plans.len(), 3);

            driver::stream_destroy(stream).unwrap();
        }
    }
}
