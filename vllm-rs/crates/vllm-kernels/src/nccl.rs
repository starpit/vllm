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

        let comm = cudarc::nccl::Comm::from_rank(stream, rank, world_size, nccl_id)
            .map_err(|e| KernelError::Other(format!("NCCL comm_init_rank failed: {e:?}")))?;

        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }

    /// Create communicators for all ranks on the current machine.
    ///
    /// This is the simplest init path: creates one communicator per CUDA device
    /// using `comm_init_all` (single-process, multi-GPU).
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
}
