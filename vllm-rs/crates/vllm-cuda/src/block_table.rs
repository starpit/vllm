// SPDX-License-Identifier: Apache-2.0
//! Persistent GPU block table with incremental updates.
//!
//! Matches Python vLLM's `BlockTables` + `StagedWriteTensor` pattern:
//! - A persistent table `[max_num_reqs, max_blocks_per_seq]` indexed by stable
//!   req_idx, updated incrementally (only new block IDs are H2D-copied).
//! - An input table `[max_graph_bs, max_blocks_per_seq]` in batch order,
//!   populated by a GPU gather kernel before each forward pass.
//! - The CUDA graph captures with the input table's pointer, so gather
//!   updates are visible during replay with zero full-table H2D copies.

use crate::alloc::RawGpuMem;
use crate::driver;
use crate::kernels;
use crate::tensor::GpuTensor;
use anyhow::Result;
use cudarc::driver::sys::CUstream;

/// Persistent GPU block table with GPU-side gather for CUDA graphs.
pub struct GpuBlockTable {
    /// `[max_num_reqs, max_blocks_per_seq]` i32 on GPU — source of truth.
    persistent: RawGpuMem,
    /// `[max_graph_bs, max_blocks_per_seq]` i32 on GPU — batch-ordered view
    /// that CUDA graphs read from (pointer baked in at capture time).
    input: RawGpuMem,
    /// `[max_num_reqs]` i32 on GPU — valid block count per row.
    num_blocks_gpu: RawGpuMem,
    /// CPU mirror of per-row block counts.
    num_blocks_cpu: Vec<i32>,
    /// `[max_graph_bs]` i32 on GPU — req_idx mapping for gather.
    req_indices_gpu: RawGpuMem,
    /// Pinned host buffer for small H2D transfers (req_indices + num_blocks).
    pinned_staging: *mut u8,
    pub max_num_reqs: usize,
    pub max_blocks_per_seq: usize,
    pub max_graph_bs: usize,
}

unsafe impl Send for GpuBlockTable {}
unsafe impl Sync for GpuBlockTable {}

impl GpuBlockTable {
    /// Allocate persistent and input block tables on GPU.
    ///
    /// # Safety
    /// Requires active CUDA context.
    pub unsafe fn new(
        max_num_reqs: usize,
        max_blocks_per_seq: usize,
        max_graph_bs: usize,
    ) -> Result<Self> {
        let persistent_bytes = max_num_reqs * max_blocks_per_seq * 4;
        let input_bytes = max_graph_bs * max_blocks_per_seq * 4;
        let num_blocks_bytes = max_num_reqs * 4;
        let req_indices_bytes = max_graph_bs * 4;

        let persistent = RawGpuMem::new(driver::mem_alloc(persistent_bytes)?, persistent_bytes);
        let input = RawGpuMem::new(driver::mem_alloc(input_bytes)?, input_bytes);
        let num_blocks_gpu =
            RawGpuMem::new(driver::mem_alloc(num_blocks_bytes)?, num_blocks_bytes);
        let req_indices_gpu =
            RawGpuMem::new(driver::mem_alloc(req_indices_bytes)?, req_indices_bytes);

        // Zero persistent and input tables.
        driver::memset_d8(persistent.ptr(), 0, persistent_bytes, std::ptr::null_mut())?;
        driver::memset_d8(input.ptr(), 0, input_bytes, std::ptr::null_mut())?;
        driver::memset_d8(num_blocks_gpu.ptr(), 0, num_blocks_bytes, std::ptr::null_mut())?;

        // Pinned staging: enough for max_num_reqs i32 (num_blocks) + max_graph_bs i32 (req_indices).
        let pinned_bytes = (max_num_reqs + max_graph_bs) * 4;
        let pinned_staging = driver::mem_alloc_host(pinned_bytes)?;

        let alloc_mb = (persistent_bytes + input_bytes + num_blocks_bytes + req_indices_bytes)
            as f64
            / (1024.0 * 1024.0);
        tracing::info!(
            "GpuBlockTable: persistent=[{max_num_reqs}, {max_blocks_per_seq}], \
             input=[{max_graph_bs}, {max_blocks_per_seq}], {alloc_mb:.1} MB"
        );

        Ok(Self {
            persistent,
            input,
            num_blocks_gpu,
            num_blocks_cpu: vec![0i32; max_num_reqs],
            req_indices_gpu,
            pinned_staging,
            max_num_reqs,
            max_blocks_per_seq,
            max_graph_bs,
        })
    }

    /// Append new block IDs for a request. Only the new IDs are H2D-copied.
    ///
    /// # Safety
    /// `req_idx` must be < `max_num_reqs`. CUDA context must be current.
    pub unsafe fn append_blocks(
        &mut self,
        req_idx: usize,
        new_block_ids: &[i32],
        stream: CUstream,
    ) -> Result<()> {
        if new_block_ids.is_empty() {
            return Ok(());
        }
        let start = self.num_blocks_cpu[req_idx] as usize;
        let end = start + new_block_ids.len();
        debug_assert!(
            end <= self.max_blocks_per_seq,
            "req_idx={req_idx}: block count {end} exceeds max_blocks_per_seq {}",
            self.max_blocks_per_seq
        );

        // H2D of just the new block IDs to persistent[req_idx, start..end].
        let offset = (req_idx * self.max_blocks_per_seq + start) * 4;
        let dst = self.persistent.ptr().add(offset);
        driver::memcpy_htod_async(
            dst,
            new_block_ids.as_ptr() as *const u8,
            new_block_ids.len() * 4,
            stream,
        )?;

        self.num_blocks_cpu[req_idx] = end as i32;
        Ok(())
    }

    /// Set all block IDs for a request (overwrite). Used for initial population.
    ///
    /// # Safety
    /// `req_idx` must be < `max_num_reqs`. CUDA context must be current.
    pub unsafe fn set_blocks(
        &mut self,
        req_idx: usize,
        block_ids: &[i32],
        stream: CUstream,
    ) -> Result<()> {
        debug_assert!(
            block_ids.len() <= self.max_blocks_per_seq,
            "req_idx={req_idx}: block count {} exceeds max_blocks_per_seq {}",
            block_ids.len(),
            self.max_blocks_per_seq
        );

        let offset = req_idx * self.max_blocks_per_seq * 4;
        let dst = self.persistent.ptr().add(offset);
        if !block_ids.is_empty() {
            driver::memcpy_htod_async(
                dst,
                block_ids.as_ptr() as *const u8,
                block_ids.len() * 4,
                stream,
            )?;
        }

        self.num_blocks_cpu[req_idx] = block_ids.len() as i32;
        Ok(())
    }

    /// Clear a request's block table entry (request finished).
    pub fn clear_req(&mut self, req_idx: usize) {
        self.num_blocks_cpu[req_idx] = 0;
    }

    /// Number of blocks currently assigned to a request.
    pub fn num_blocks(&self, req_idx: usize) -> usize {
        self.num_blocks_cpu[req_idx] as usize
    }

    /// GPU-side gather: copy used blocks from persistent table to input table
    /// in batch order. Uploads `req_indices` and `num_blocks_cpu`, then
    /// launches the gather kernel.
    ///
    /// # Safety
    /// CUDA context must be current. `batch_req_indices` maps batch position
    /// to req_idx (length = `batch_size`).
    pub unsafe fn gather(
        &self,
        batch_req_indices: &[usize],
        batch_size: usize,
        stream: CUstream,
    ) -> Result<()> {
        if batch_size == 0 {
            return Ok(());
        }
        debug_assert!(batch_size <= self.max_graph_bs);

        // Stage req_indices into pinned buffer.
        let ri_pinned = self.pinned_staging as *mut i32;
        for (i, &idx) in batch_req_indices.iter().enumerate().take(batch_size) {
            *ri_pinned.add(i) = idx as i32;
        }
        // H2D req_indices.
        driver::memcpy_htod_async(
            self.req_indices_gpu.ptr(),
            ri_pinned as *const u8,
            batch_size * 4,
            stream,
        )?;

        // Stage num_blocks into pinned buffer (after req_indices section).
        let nb_pinned = self.pinned_staging.add(self.max_graph_bs * 4) as *mut i32;
        std::ptr::copy_nonoverlapping(
            self.num_blocks_cpu.as_ptr(),
            nb_pinned,
            self.max_num_reqs,
        );
        // H2D num_blocks.
        driver::memcpy_htod_async(
            self.num_blocks_gpu.ptr(),
            nb_pinned as *const u8,
            self.max_num_reqs * 4,
            stream,
        )?;

        // Launch gather kernel.
        kernels::gather_block_table_gpu(
            self.persistent.ptr(),
            self.input.ptr(),
            self.num_blocks_gpu.ptr(),
            self.req_indices_gpu.ptr(),
            self.max_blocks_per_seq,
            self.max_blocks_per_seq,
            batch_size,
            stream,
        );

        Ok(())
    }

    /// Raw pointer to the input table (for CUDA graph capture).
    pub fn input_ptr(&self) -> *mut u8 {
        self.input.ptr()
    }

    /// View of the input table as a GpuTensor.
    ///
    /// # Safety
    /// The returned tensor borrows the input buffer — do not outlive `self`.
    pub unsafe fn input_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.input.ptr(),
            &[batch_size, self.max_blocks_per_seq],
            crate::dtype::DType::I32,
        )
    }
}

impl Drop for GpuBlockTable {
    fn drop(&mut self) {
        // RawGpuMem handles persistent, input, num_blocks_gpu, req_indices_gpu.
        // We just need to free the pinned staging buffer.
        if !self.pinned_staging.is_null() {
            unsafe {
                let _ = driver::mem_free_host(self.pinned_staging);
            }
        }
    }
}
