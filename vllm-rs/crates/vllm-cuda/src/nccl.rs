// SPDX-License-Identifier: Apache-2.0
//! NCCL collective communication for `GpuTensor`.
//!
//! Wraps cudarc's low-level NCCL FFI (raw pointers) to work with our `GpuTensor`
//! type. One `NcclGroup` per GPU rank — created during TP init, stored in model
//! layers for all-reduce/all-gather calls in the forward pass.

use std::cell::Cell;
use std::ffi::c_void;
use std::mem::MaybeUninit;

use anyhow::Result;
use cudarc::driver::sys::CUstream;
use cudarc::nccl::{result as nccl_result, sys as nccl_sys};

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// Piecewise graph capture: suppress NCCL collectives
// ---------------------------------------------------------------------------

thread_local! {
    /// When true, NCCL collectives (all-reduce, all-gather) become no-ops.
    /// Set during piecewise CUDA graph capture/replay so NCCL operations are
    /// NOT baked into graphs — they run eagerly between graph pieces instead.
    static SUPPRESS_NCCL: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that suppresses NCCL collectives for the duration of its lifetime.
/// Used by piecewise graph capture and replay.
pub struct SuppressNcclGuard {
    prev: bool,
}

impl Default for SuppressNcclGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SuppressNcclGuard {
    pub fn new() -> Self {
        let prev = SUPPRESS_NCCL.with(|c| c.replace(true));
        Self { prev }
    }
}

impl Drop for SuppressNcclGuard {
    fn drop(&mut self) {
        SUPPRESS_NCCL.with(|c| c.set(self.prev));
    }
}

/// Returns true if NCCL collectives should be suppressed (during piecewise graph capture/replay).
pub fn is_nccl_suppressed() -> bool {
    SUPPRESS_NCCL.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// NcclId — wrapper around ncclUniqueId
// ---------------------------------------------------------------------------

/// NCCL unique ID for communicator creation. Generated on rank 0, shared to all.
#[derive(Clone, Copy)]
pub struct NcclId(nccl_sys::ncclUniqueId);

// Raw bytes — safe to send across threads.
unsafe impl Send for NcclId {}
unsafe impl Sync for NcclId {}

impl NcclId {
    /// Generate a new unique NCCL ID (call on rank 0 only).
    pub fn new() -> Result<Self> {
        let id = nccl_result::get_uniqueid()
            .map_err(|e| anyhow::anyhow!("ncclGetUniqueId failed: {:?}", e))?;
        Ok(Self(id))
    }

    /// Create from raw bytes (received from rank 0).
    pub fn from_raw(internal: [core::ffi::c_char; 128]) -> Self {
        Self(nccl_sys::ncclUniqueId { internal })
    }

    /// Get raw bytes (for sending to other ranks).
    pub fn raw(&self) -> &[core::ffi::c_char; 128] {
        &self.0.internal
    }
}

// ---------------------------------------------------------------------------
// NcclGroup — one per GPU rank
// ---------------------------------------------------------------------------

/// NCCL communicator for a single GPU rank.
///
/// Wraps `ncclComm_t` and operates on raw GPU pointers from `GpuTensor`.
/// All collective operations are enqueued on the provided CUDA stream
/// (non-blocking on the host).
pub struct NcclGroup {
    comm: nccl_sys::ncclComm_t,
    rank: usize,
    world_size: usize,
    stream: CUstream,
}

// NCCL comms are thread-safe for enqueue operations.
unsafe impl Send for NcclGroup {}
unsafe impl Sync for NcclGroup {}

impl NcclGroup {
    /// Create a new NCCL communicator for this rank.
    ///
    /// Must be called from the thread that owns the CUDA context for this GPU.
    /// All ranks must call this concurrently with the same `id` and `world_size`.
    pub fn new(rank: usize, world_size: usize, id: NcclId, stream: CUstream) -> Result<Self> {
        let mut comm = MaybeUninit::uninit();
        unsafe {
            nccl_result::comm_init_rank(comm.as_mut_ptr(), world_size as i32, id.0, rank as i32)
                .map_err(|e| anyhow::anyhow!("ncclCommInitRank failed for rank {rank}: {:?}", e))?;
        }
        let comm = unsafe { comm.assume_init() };
        Ok(Self {
            comm,
            rank,
            world_size,
            stream,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// In-place all-reduce (sum) on a GpuTensor.
    ///
    /// The tensor is modified in place. All ranks must call with tensors of
    /// the same shape and dtype. Non-blocking on the host.
    pub unsafe fn all_reduce_inplace(&self, tensor: GpuTensor) -> Result<()> {
        // During piecewise graph capture/replay, skip NCCL — it runs eagerly
        // between graph pieces instead.
        if is_nccl_suppressed() {
            return Ok(());
        }
        let numel = tensor.numel();
        let nccl_dtype = gpu_dtype_to_nccl(tensor.dtype())?;
        let ptr = tensor.raw_ptr() as *mut c_void;

        nccl_result::all_reduce(
            ptr as *const c_void,
            ptr,
            numel,
            nccl_dtype,
            nccl_sys::ncclRedOp_t::ncclSum,
            self.comm,
            self.stream as nccl_sys::cudaStream_t,
        )
        .map_err(|e| anyhow::anyhow!("ncclAllReduce failed: {:?}", e))?;
        Ok(())
    }

    /// Point-to-point send to `peer` rank. Non-blocking on the host.
    ///
    /// Enqueued on the comm's CUDA stream. Matches Python's `ncclSend` —
    /// no `ncclGroupStart/End` wrapping (each tensor gets its own call).
    pub unsafe fn send(&self, tensor: GpuTensor, peer: usize) -> Result<()> {
        let numel = tensor.numel();
        let nccl_dtype = gpu_dtype_to_nccl(tensor.dtype())?;
        let ptr = tensor.raw_ptr() as *const c_void;

        nccl_result::send(
            ptr,
            numel,
            nccl_dtype,
            peer as i32,
            self.comm,
            self.stream as nccl_sys::cudaStream_t,
        )
        .map_err(|e| anyhow::anyhow!("ncclSend to peer {peer} failed: {:?}", e))?;
        Ok(())
    }

    /// Point-to-point recv from `peer` rank into a pre-allocated buffer.
    /// Non-blocking on the host.
    ///
    /// The tensor must be pre-allocated with the correct shape and dtype.
    /// Enqueued on the comm's CUDA stream.
    pub unsafe fn recv(&self, tensor: GpuTensor, peer: usize) -> Result<()> {
        let numel = tensor.numel();
        let nccl_dtype = gpu_dtype_to_nccl(tensor.dtype())?;
        let ptr = tensor.raw_ptr() as *mut c_void;

        nccl_result::recv(
            ptr,
            numel,
            nccl_dtype,
            peer as i32,
            self.comm,
            self.stream as nccl_sys::cudaStream_t,
        )
        .map_err(|e| anyhow::anyhow!("ncclRecv from peer {peer} failed: {:?}", e))?;
        Ok(())
    }

    /// In-place broadcast from `root` to all ranks. Non-blocking on the host.
    ///
    /// The root rank's tensor data is broadcast to all other ranks.
    /// All ranks must call with the same shape, dtype, and root.
    pub unsafe fn broadcast_inplace(&self, tensor: GpuTensor, root: usize) -> Result<()> {
        let numel = tensor.numel();
        let nccl_dtype = gpu_dtype_to_nccl(tensor.dtype())?;
        let ptr = tensor.raw_ptr() as *mut c_void;

        nccl_result::broadcast(
            ptr as *const c_void,
            ptr,
            numel,
            nccl_dtype,
            root as i32,
            self.comm,
            self.stream as nccl_sys::cudaStream_t,
        )
        .map_err(|e| anyhow::anyhow!("ncclBroadcast from root {root} failed: {:?}", e))?;
        Ok(())
    }

    /// Get the CUDA stream associated with this communicator.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// All-gather along dim=0: each rank contributes `tensor` (same shape),
    /// output is `[world_size * dim0, ...]`.
    ///
    /// Allocates output from `alloc`. Non-blocking on the host.
    pub unsafe fn all_gather(
        &self,
        tensor: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        let numel = tensor.numel();
        let nccl_dtype =
            gpu_dtype_to_nccl(tensor.dtype()).expect("unsupported dtype for NCCL all_gather");

        // Output shape: dim0 * world_size, rest unchanged.
        let mut out_shape: Vec<usize> = (0..tensor.ndim()).map(|i| tensor.dim(i)).collect();
        out_shape[0] *= self.world_size;

        let out = alloc.alloc_tensor(&out_shape, tensor.dtype());

        // During piecewise graph capture/replay, skip the NCCL call.
        if !is_nccl_suppressed() {
            let out_gpu = out.as_gpu_tensor();
            nccl_result::all_gather(
                tensor.raw_ptr() as *const c_void,
                out_gpu.raw_ptr() as *mut c_void,
                numel,
                nccl_dtype,
                self.comm,
                self.stream as nccl_sys::cudaStream_t,
            )
            .expect("ncclAllGather failed");
        }

        out
    }

    /// All-gather along the last dimension of a 2D tensor.
    ///
    /// Input `[N, S]` per rank → output `[N, world_size * S]`.
    ///
    /// NCCL only supports contiguous (dim=0) gather, so this does:
    /// 1. NCCL all-gather → `[world_size * N, S]` (temp buffer)
    /// 2. Rearrange to `[N, world_size * S]` via a copy kernel
    ///
    /// Matches Python vLLM's `tensor_model_parallel_all_gather(logits)`
    /// which gathers along dim=-1 using movedim + contiguous.
    pub unsafe fn all_gather_last_dim(
        &self,
        tensor: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        if tensor.ndim() != 2 {
            panic!(
                "all_gather_last_dim requires 2D tensor, got {}D",
                tensor.ndim()
            );
        }
        let n = tensor.dim(0); // num_reqs
        let s = tensor.dim(1); // vocab_shard = vocab / world_size

        // Step 1: NCCL all-gather along dim=0 into temp buffer.
        let temp = self.all_gather(tensor, alloc);

        // Step 2: Rearrange [world_size * N, S] → [N, world_size * S].
        let vocab = self.world_size * s;
        let out = alloc.alloc_tensor(&[n, vocab], temp.as_gpu_tensor().dtype());

        let elem_bytes = temp.as_gpu_tensor().dtype().size_bytes();
        match elem_bytes {
            2 => gather_last_dim_f16(
                temp.as_gpu_tensor().raw_ptr(),
                out.as_gpu_tensor().raw_ptr() as *mut u8,
                n as i32,
                s as i32,
                self.world_size as i32,
                self.stream,
            ),
            4 => gather_last_dim_f32(
                temp.as_gpu_tensor().raw_ptr(),
                out.as_gpu_tensor().raw_ptr() as *mut u8,
                n as i32,
                s as i32,
                self.world_size as i32,
                self.stream,
            ),
            _ => panic!("all_gather_last_dim: unsupported dtype size {elem_bytes}"),
        }

        out
    }
}

// FFI for gather_last_dim rearrangement kernel.
unsafe extern "C" {
    fn gather_last_dim_f16(
        src: *const u8,
        dst: *mut u8,
        n: i32,
        s: i32,
        world_size: i32,
        stream: CUstream,
    );
    fn gather_last_dim_f32(
        src: *const u8,
        dst: *mut u8,
        n: i32,
        s: i32,
        world_size: i32,
        stream: CUstream,
    );
}

impl Drop for NcclGroup {
    fn drop(&mut self) {
        unsafe {
            let _ = nccl_result::comm_abort(self.comm);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gpu_dtype_to_nccl(dtype: DType) -> Result<nccl_sys::ncclDataType_t> {
    match dtype {
        DType::F16 => Ok(nccl_sys::ncclDataType_t::ncclFloat16),
        DType::BF16 => Ok(nccl_sys::ncclDataType_t::ncclBfloat16),
        DType::F32 => Ok(nccl_sys::ncclDataType_t::ncclFloat32),
        DType::I32 => Ok(nccl_sys::ncclDataType_t::ncclInt32),
        DType::I64 => Ok(nccl_sys::ncclDataType_t::ncclInt64),
        DType::U32 => Ok(nccl_sys::ncclDataType_t::ncclUint32),
        DType::U8 => Ok(nccl_sys::ncclDataType_t::ncclUint8),
        DType::Fp8E4m3 => Ok(nccl_sys::ncclDataType_t::ncclUint8), // FP8 as raw bytes
    }
}
