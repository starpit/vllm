// SPDX-License-Identifier: Apache-2.0
//! NCCL-based multi-GPU collective communication.
//!
//! Wraps `cudarc::nccl` to provide candle-tensor-level all-reduce and
//! all-gather operations for tensor parallelism.
//!
//! Port of: `vllm/distributed/parallel_state.py` (communication ops)

use candle_core::cuda_backend::CudaDType;
use candle_core::{DType, Device, Storage, Tensor};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// NcclProcessGroup
// ---------------------------------------------------------------------------

/// NCCL communicator wrapping `cudarc::nccl::Comm` for candle tensor collectives.
///
/// Each GPU rank in a tensor-parallel group owns one `NcclProcessGroup`.
/// The communicator is created once during init and shared (via `Arc`) with
/// all parallel linear layers that need collective communication.
pub struct NcclProcessGroup {
    comm: cudarc::nccl::Comm,
    rank: usize,
    world_size: usize,
}

// Safety: cudarc::nccl::Comm is internally synchronized via NCCL and CUDA streams.
// Each NcclProcessGroup is pinned to a single GPU thread.
unsafe impl Send for NcclProcessGroup {}
unsafe impl Sync for NcclProcessGroup {}

impl NcclProcessGroup {
    /// Create a new NCCL communicator for the given rank.
    ///
    /// All ranks must call this with the same `nccl_id` (generated once by
    /// rank 0 via `cudarc::nccl::Id::new()`).
    ///
    /// The `device` must be a `Device::Cuda(ordinal)` matching this rank.
    pub fn new(
        rank: usize,
        world_size: usize,
        nccl_id: cudarc::nccl::Id,
        device: &Device,
    ) -> KernelResult<Self> {
        let stream = match device {
            Device::Cuda(cuda_dev) => cuda_dev.cuda_stream(),
            _ => {
                return Err(KernelError::Other(
                    "NCCL requires a CUDA device".to_string(),
                ));
            }
        };

        let comm =
            cudarc::nccl::Comm::from_rank(stream, rank, world_size, nccl_id).map_err(|e| {
                let mut msg = format!(
                    "NCCL comm_init_rank failed (rank={rank}, world_size={world_size}): {e:?}"
                );
                #[cfg(target_os = "linux")]
                {
                    // NCCL uses /dev/shm for inter-process communication.
                    // Check if it exists and has space.
                    let shm = std::path::Path::new("/dev/shm");
                    if !shm.exists() {
                        msg.push_str(
                            "\n  hint: /dev/shm does not exist — NCCL requires shared memory",
                        );
                    } else if let Ok(entries) = std::fs::read_dir(shm) {
                        let nccl_count = entries
                            .flatten()
                            .filter(|e| {
                                e.file_name()
                                    .to_str()
                                    .is_some_and(|n| n.starts_with("nccl-"))
                            })
                            .count();
                        if nccl_count > 0 {
                            msg.push_str(&format!(
                                "\n  hint: found {nccl_count} stale nccl-* files in /dev/shm — \
                                 try removing them: rm /dev/shm/nccl-*"
                            ));
                        }
                    }
                }
                KernelError::Other(msg)
            })?;

        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }

    /// Generate a new NCCL unique ID for bootstrapping communicator creation.
    ///
    /// Call this once on rank 0, then distribute the ID to all ranks via
    /// TCP rendezvous or shared memory.
    pub fn generate_id() -> KernelResult<cudarc::nccl::Id> {
        cudarc::nccl::Id::new()
            .map_err(|e| KernelError::Other(format!("failed to create NCCL ID: {e:?}")))
    }

    /// Create per-rank communicators for all local devices using `comm_init_rank`.
    ///
    /// Unlike `from_devices` (which uses `comm_init_all` and ties comms to
    /// the calling thread), this creates independent communicators that work
    /// correctly when used from separate threads (e.g. tokio worker tasks).
    ///
    /// `base_rank` is the global rank offset for the first device.
    /// `global_world_size` is the total number of GPUs across all nodes.
    /// Create per-rank communicators for local CUDA devices.
    ///
    /// `cuda_ordinals` are the CUDA device indices (e.g. `[0, 1]`).
    /// Each comm is created on its own thread via `comm_init_rank` to avoid
    /// deadlock (the call is collective) and ensure the correct CUDA context.
    ///
    /// Unlike `from_devices` (which uses `comm_init_all` and ties comms to
    /// the calling thread), these communicators work correctly when used
    /// from separate threads (e.g. worker threads in ThreadPoolExecutor).
    pub fn from_device_ordinals(
        cuda_ordinals: &[usize],
        base_rank: usize,
        global_world_size: usize,
    ) -> KernelResult<Vec<Self>> {
        let nccl_id = Self::generate_id()?;

        // comm_init_rank is a collective — all ranks must call it concurrently.
        // Spawn each rank on its own thread with a fresh CUDA device to ensure
        // the correct CUDA context is active.
        let handles: Vec<_> = cuda_ordinals
            .iter()
            .enumerate()
            .map(|(local_rank, &ordinal)| {
                let global_rank = base_rank + local_rank;
                std::thread::spawn(move || {
                    let device = Device::new_cuda(ordinal).map_err(|e| {
                        KernelError::Other(format!("failed to create CUDA device {ordinal}: {e}"))
                    })?;
                    Self::new(global_rank, global_world_size, nccl_id, &device)
                })
            })
            .collect();

        let mut groups = Vec::with_capacity(handles.len());
        for handle in handles {
            let group: KernelResult<Self> = handle
                .join()
                .map_err(|_| KernelError::Other("NCCL init thread panicked".to_string()))?;
            groups.push(group?);
        }
        Ok(groups)
    }

    /// Create communicators for all ranks on the current machine.
    ///
    /// This is the simplest init path: creates one communicator per CUDA device
    /// using `comm_init_all` (single-process, multi-GPU).
    ///
    /// **Note**: Communicators created this way may not work correctly when
    /// used from different threads. Prefer `from_devices_per_rank` for
    /// multi-threaded executors.
    pub fn from_devices(devices: &[Device]) -> KernelResult<Vec<Self>> {
        let streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>> = devices
            .iter()
            .map(|d| match d {
                Device::Cuda(cuda_dev) => Ok(cuda_dev.cuda_stream()),
                _ => Err(KernelError::Other("NCCL requires CUDA devices".to_string())),
            })
            .collect::<KernelResult<Vec<_>>>()?;

        let world_size = streams.len();
        let comms = cudarc::nccl::Comm::from_devices(streams)
            .map_err(|e| KernelError::Other(format!("NCCL comm_init_all failed: {e:?}")))?;

        Ok(comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| Self {
                comm,
                rank,
                world_size,
            })
            .collect())
    }

    /// This rank's index (0-based).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks in this group.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// All-reduce (sum) a candle tensor in-place across all ranks.
    ///
    /// The input tensor must be contiguous and on the correct CUDA device.
    /// Returns a new tensor with the reduced values.
    pub fn all_reduce(&self, tensor: &Tensor) -> KernelResult<Tensor> {
        let tensor = tensor
            .contiguous()
            .map_err(|e| KernelError::Other(format!("contiguous failed: {e}")))?;
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();

        let (storage, layout) = tensor.storage_and_layout();
        let offset = layout.start_offset();

        match &*storage {
            Storage::Cuda(cuda_storage) => match dtype {
                DType::F32 => self.all_reduce_typed::<f32>(cuda_storage, offset, &shape),
                DType::F16 => self.all_reduce_typed::<half::f16>(cuda_storage, offset, &shape),
                DType::BF16 => self.all_reduce_typed::<half::bf16>(cuda_storage, offset, &shape),
                _ => Err(KernelError::Other(format!(
                    "NCCL all_reduce unsupported dtype: {dtype:?}"
                ))),
            },
            _ => Err(KernelError::Other(
                "NCCL all_reduce requires CUDA tensor".to_string(),
            )),
        }
    }

    /// All-gather a candle tensor along dimension 0 across all ranks.
    ///
    /// Each rank contributes its local tensor; the output has the local
    /// tensors concatenated along the first dimension (world_size * local_dim0).
    pub fn all_gather(&self, tensor: &Tensor, dim: usize) -> KernelResult<Tensor> {
        if dim != 0 {
            return Err(KernelError::Other(
                "NCCL all_gather only supports dim=0 currently".to_string(),
            ));
        }

        let tensor = tensor
            .contiguous()
            .map_err(|e| KernelError::Other(format!("contiguous failed: {e}")))?;
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();

        let (storage, layout) = tensor.storage_and_layout();
        let offset = layout.start_offset();

        match &*storage {
            Storage::Cuda(cuda_storage) => match dtype {
                DType::F32 => self.all_gather_typed::<f32>(cuda_storage, offset, &shape),
                DType::F16 => self.all_gather_typed::<half::f16>(cuda_storage, offset, &shape),
                DType::BF16 => self.all_gather_typed::<half::bf16>(cuda_storage, offset, &shape),
                _ => Err(KernelError::Other(format!(
                    "NCCL all_gather unsupported dtype: {dtype:?}"
                ))),
            },
            _ => Err(KernelError::Other(
                "NCCL all_gather requires CUDA tensor".to_string(),
            )),
        }
    }

    /// Broadcast a byte buffer from `root` rank to all other ranks via NCCL.
    ///
    /// Uses a single NCCL broadcast with a fixed-size buffer. The first 4
    /// bytes encode the payload length (u32 LE), followed by the payload.
    /// Max payload: 4MB (sufficient for serialized SchedulerOutput).
    ///
    /// All ranks must call this simultaneously (it's a collective).
    ///
    /// - On `root`: `data` is the payload to send.
    /// - On non-root: `data` is ignored; the returned `Vec<u8>` is the payload.
    pub fn broadcast_bytes(&self, data: &[u8], root: usize) -> KernelResult<Vec<u8>> {
        // Fixed buffer: 4 bytes length header + up to 4MB payload.
        const MAX_PAYLOAD: usize = 4 * 1024 * 1024;
        const HEADER_SIZE: usize = 4; // u32 length
        let buf_bytes = HEADER_SIZE + MAX_PAYLOAD;
        let num_u32 = buf_bytes / 4;

        if data.len() > MAX_PAYLOAD {
            return Err(KernelError::Other(format!(
                "broadcast_bytes: data too large ({} bytes, max {})",
                data.len(),
                MAX_PAYLOAD
            )));
        }

        let stream = self.comm.stream();

        // Allocate fixed-size GPU buffer.
        let mut buf = stream
            .alloc_zeros::<u32>(num_u32)
            .map_err(|e| KernelError::Other(format!("alloc broadcast buf failed: {e}")))?;

        // Root: pack length + payload into the buffer.
        if self.rank == root {
            let len = data.len() as u32;
            let mut host_buf = vec![0u32; num_u32];
            // Write length as first u32.
            host_buf[0] = len;
            // Copy payload bytes after the header.
            if !data.is_empty() {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        (host_buf.as_mut_ptr() as *mut u8).add(HEADER_SIZE),
                        MAX_PAYLOAD,
                    )
                };
                dst[..data.len()].copy_from_slice(data);
            }
            stream
                .memcpy_htod(&host_buf, &mut buf)
                .map_err(|e| KernelError::Other(format!("memcpy H2D failed: {e}")))?;
        }

        // Single NCCL broadcast — all ranks participate.
        self.comm
            .broadcast_in_place(&mut buf, root as i32)
            .map_err(|e| KernelError::Other(format!("NCCL broadcast failed: {e:?}")))?;

        // All ranks: read buffer back to host.
        let mut host_buf = vec![0u32; num_u32];
        stream
            .memcpy_dtoh(&buf, &mut host_buf)
            .map_err(|e| KernelError::Other(format!("memcpy D2H failed: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| KernelError::Other(format!("sync failed: {e}")))?;

        // Extract length and payload.
        let len = host_buf[0] as usize;
        let payload = unsafe {
            let ptr = (host_buf.as_ptr() as *const u8).add(HEADER_SIZE);
            std::slice::from_raw_parts(ptr, len)
        };
        Ok(payload.to_vec())
    }

    // -----------------------------------------------------------------------
    // Internal typed helpers
    // -----------------------------------------------------------------------

    fn all_reduce_typed<T>(
        &self,
        cuda_storage: &candle_core::CudaStorage,
        offset: usize,
        shape: &candle_core::Shape,
    ) -> KernelResult<Tensor>
    where
        T: CudaDType
            + cudarc::nccl::NcclType
            + cudarc::driver::DeviceRepr
            + cudarc::driver::ValidAsZeroBits,
    {
        let src_slice = T::as_cuda_slice(cuda_storage)
            .map_err(|e| KernelError::Other(format!("as_cuda_slice failed: {e}")))?;
        let src = src_slice.slice(offset..offset + shape.elem_count());

        let num_elems = shape.elem_count();
        let mut dst = self
            .comm
            .stream()
            .alloc_zeros::<T>(num_elems)
            .map_err(|e| KernelError::Other(format!("alloc_zeros failed: {e}")))?;

        self.comm
            .all_reduce(&src, &mut dst, &cudarc::nccl::ReduceOp::Sum)
            .map_err(|e| KernelError::Other(format!("NCCL all_reduce failed: {e:?}")))?;

        let out_storage = T::wrap_cuda_slice(dst, cuda_storage.device.clone());
        let out = Tensor::from_storage(
            Storage::Cuda(out_storage),
            shape.clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }

    fn all_gather_typed<T>(
        &self,
        cuda_storage: &candle_core::CudaStorage,
        offset: usize,
        shape: &candle_core::Shape,
    ) -> KernelResult<Tensor>
    where
        T: CudaDType
            + cudarc::nccl::NcclType
            + cudarc::driver::DeviceRepr
            + cudarc::driver::ValidAsZeroBits,
    {
        let src_slice = T::as_cuda_slice(cuda_storage)
            .map_err(|e| KernelError::Other(format!("as_cuda_slice failed: {e}")))?;
        let src = src_slice.slice(offset..offset + shape.elem_count());

        let num_elems = shape.elem_count();
        let total_elems = num_elems * self.world_size;
        let mut dst = self
            .comm
            .stream()
            .alloc_zeros::<T>(total_elems)
            .map_err(|e| KernelError::Other(format!("alloc_zeros failed: {e}")))?;

        self.comm
            .all_gather(&src, &mut dst)
            .map_err(|e| KernelError::Other(format!("NCCL all_gather failed: {e:?}")))?;

        // Output shape: [world_size * dim0, dim1, dim2, ...]
        let mut out_dims = shape.dims().to_vec();
        out_dims[0] *= self.world_size;
        let out_shape = candle_core::Shape::from_dims(&out_dims);

        let out_storage = T::wrap_cuda_slice(dst, cuda_storage.device.clone());
        let out = Tensor::from_storage(
            Storage::Cuda(out_storage),
            out_shape,
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }
}

impl std::fmt::Debug for NcclProcessGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclProcessGroup")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish()
    }
}

impl vllm_model::process_group::ProcessGroup for NcclProcessGroup {
    fn all_reduce(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        NcclProcessGroup::all_reduce(self, tensor)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn all_gather(&self, tensor: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
        NcclProcessGroup::all_gather(self, tensor, dim)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn broadcast_bytes(&self, data: &[u8], root: usize) -> candle_core::Result<Vec<u8>> {
        NcclProcessGroup::broadcast_bytes(self, data, root)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests (require CUDA + NCCL at runtime)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test NCCL all-reduce with 2 communicators on available GPUs.
    ///
    /// Requires at least 2 CUDA devices. Run with:
    ///   cargo test -p vllm-kernels --features nccl -- test_nccl_all_reduce --test-threads=1
    #[test]
    #[ignore] // Requires multi-GPU hardware
    fn test_nccl_all_reduce() {
        let n_devices = cudarc::driver::CudaContext::device_count().unwrap() as usize;
        if n_devices < 2 {
            eprintln!("Skipping test_nccl_all_reduce: need >= 2 CUDA devices, found {n_devices}");
            return;
        }

        let dev0 = Device::cuda_if_available(0).unwrap();
        let dev1 = Device::new_cuda(1).unwrap();

        let groups = NcclProcessGroup::from_devices(&[dev0.clone(), dev1.clone()]).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].rank(), 0);
        assert_eq!(groups[1].rank(), 1);

        // Rank 0: [1, 2, 3], Rank 1: [4, 5, 6]
        let t0 = Tensor::new(&[1.0f32, 2.0, 3.0], &dev0).unwrap();
        let t1 = Tensor::new(&[4.0f32, 5.0, 6.0], &dev1).unwrap();

        // Use threads because NCCL collectives must be called concurrently.
        let g0 = &groups[0];
        let g1 = &groups[1];

        // NCCL requires concurrent calls — use group API.
        cudarc::nccl::group_start().unwrap();
        let r0 = g0.all_reduce(&t0).unwrap();
        let r1 = g1.all_reduce(&t1).unwrap();
        cudarc::nccl::group_end().unwrap();

        let v0 = r0.to_vec1::<f32>().unwrap();
        let v1 = r1.to_vec1::<f32>().unwrap();

        assert_eq!(v0, vec![5.0, 7.0, 9.0]);
        assert_eq!(v1, vec![5.0, 7.0, 9.0]);
    }

    /// Test NCCL all-gather with 2 communicators.
    #[test]
    #[ignore] // Requires multi-GPU hardware
    fn test_nccl_all_gather() {
        let n_devices = cudarc::driver::CudaContext::device_count().unwrap() as usize;
        if n_devices < 2 {
            eprintln!("Skipping test_nccl_all_gather: need >= 2 CUDA devices, found {n_devices}");
            return;
        }

        let dev0 = Device::cuda_if_available(0).unwrap();
        let dev1 = Device::new_cuda(1).unwrap();

        let groups = NcclProcessGroup::from_devices(&[dev0.clone(), dev1.clone()]).unwrap();

        // Rank 0: [1, 2], Rank 1: [3, 4]
        let t0 = Tensor::new(&[1.0f32, 2.0], &dev0).unwrap();
        let t1 = Tensor::new(&[3.0f32, 4.0], &dev1).unwrap();

        cudarc::nccl::group_start().unwrap();
        let r0 = groups[0].all_gather(&t0, 0).unwrap();
        let r1 = groups[1].all_gather(&t1, 0).unwrap();
        cudarc::nccl::group_end().unwrap();

        let v0 = r0.to_vec1::<f32>().unwrap();
        let v1 = r1.to_vec1::<f32>().unwrap();

        // All-gather concatenates: [rank0_data, rank1_data]
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Test NCCL all-reduce with BF16 tensors.
    #[test]
    #[ignore] // Requires multi-GPU hardware
    fn test_nccl_all_reduce_bf16() {
        let n_devices = cudarc::driver::CudaContext::device_count().unwrap() as usize;
        if n_devices < 2 {
            return;
        }

        let dev0 = Device::cuda_if_available(0).unwrap();
        let dev1 = Device::new_cuda(1).unwrap();

        let groups = NcclProcessGroup::from_devices(&[dev0.clone(), dev1.clone()]).unwrap();

        let t0 = Tensor::new(&[1.0f32, 2.0, 3.0], &dev0)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let t1 = Tensor::new(&[4.0f32, 5.0, 6.0], &dev1)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        cudarc::nccl::group_start().unwrap();
        let r0 = groups[0].all_reduce(&t0).unwrap();
        let r1 = groups[1].all_reduce(&t1).unwrap();
        cudarc::nccl::group_end().unwrap();

        let v0 = r0.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        let v1 = r1.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

        assert!((v0[0] - 5.0).abs() < 0.1);
        assert!((v0[1] - 7.0).abs() < 0.1);
        assert!((v1[2] - 9.0).abs() < 0.1);
    }

    /// Test NCCL all-reduce from separate OS threads using `ncclCommInitRank`.
    ///
    /// This matches the real executor pattern: each rank's communicator is
    /// created on its own thread (via `NcclProcessGroup::new` with a shared ID),
    /// and all-reduce is called independently from each thread.
    #[test]
    #[ignore] // Requires multi-GPU hardware
    fn test_nccl_all_reduce_multithreaded() {
        let n_devices = cudarc::driver::CudaContext::device_count().unwrap() as usize;
        if n_devices < 2 {
            return;
        }

        // Generate a shared NCCL ID on the main thread.
        let nccl_id = cudarc::nccl::Id::new().unwrap();

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        let b0 = barrier.clone();
        let id0 = nccl_id;
        let handle0 = std::thread::spawn(move || {
            let dev = Device::cuda_if_available(0).unwrap();
            let group = NcclProcessGroup::new(0, 2, id0, &dev).unwrap();
            let t = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
            b0.wait();
            group.all_reduce(&t).unwrap().to_vec1::<f32>().unwrap()
        });

        let b1 = barrier.clone();
        let id1 = nccl_id;
        let handle1 = std::thread::spawn(move || {
            let dev = Device::new_cuda(1).unwrap();
            let group = NcclProcessGroup::new(1, 2, id1, &dev).unwrap();
            let t = Tensor::new(&[4.0f32, 5.0, 6.0], &dev).unwrap();
            b1.wait();
            group.all_reduce(&t).unwrap().to_vec1::<f32>().unwrap()
        });

        let v0 = handle0.join().unwrap();
        let v1 = handle1.join().unwrap();

        assert_eq!(v0, vec![5.0, 7.0, 9.0]);
        assert_eq!(v1, vec![5.0, 7.0, 9.0]);
    }
}
