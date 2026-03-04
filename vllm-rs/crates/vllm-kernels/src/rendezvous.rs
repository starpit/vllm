// SPDX-License-Identifier: Apache-2.0
//! TCP rendezvous for multi-node NCCL initialization.
//!
//! Rank 0 (master) generates an NCCL unique ID, listens on a TCP port, and
//! sends the 128-byte ID to all connecting worker ranks. Worker ranks connect
//! to the master and receive the ID.
//!
//! This is the standard bootstrap pattern used by PyTorch/NCCL distributed
//! training (`torch.distributed.init_process_group(init_method="tcp://...")`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::error::{KernelError, KernelResult};

/// Size of an NCCL unique ID in bytes.
const NCCL_ID_BYTES: usize = 128;

/// Master rendezvous: generate an NCCL ID and distribute it to all ranks.
///
/// Binds to `0.0.0.0:{port}`, accepts `world_size - 1` connections, and
/// sends the raw NCCL unique ID bytes to each.
///
/// Returns the generated `cudarc::nccl::Id`.
pub fn rendezvous_master(port: u16, world_size: usize) -> KernelResult<cudarc::nccl::Id> {
    let id = cudarc::nccl::Id::new()
        .map_err(|e| KernelError::Other(format!("failed to create NCCL ID: {e:?}")))?;

    let id_bytes: &[core::ffi::c_char; NCCL_ID_BYTES] = id.internal();
    // Transmute to u8 for TCP send (c_char may be i8 or u8 depending on platform).
    let bytes: &[u8; NCCL_ID_BYTES] = unsafe {
        &*(id_bytes as *const [core::ffi::c_char; NCCL_ID_BYTES] as *const [u8; NCCL_ID_BYTES])
    };

    let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
        .map_err(|e| KernelError::Other(format!("rendezvous bind failed: {e}")))?;

    tracing::info!(
        "NCCL rendezvous master listening on port {port}, waiting for {} workers",
        world_size - 1
    );

    for i in 0..(world_size - 1) {
        let (mut stream, addr) = listener
            .accept()
            .map_err(|e| KernelError::Other(format!("rendezvous accept failed: {e}")))?;

        stream
            .write_all(bytes)
            .map_err(|e| KernelError::Other(format!("rendezvous send failed: {e}")))?;

        tracing::info!(
            "NCCL rendezvous: sent ID to worker {}/{} from {addr}",
            i + 1,
            world_size - 1
        );
    }

    Ok(id)
}

/// Worker rendezvous: connect to master and receive the NCCL unique ID.
///
/// Connects to `master_addr:port` and reads the 128-byte NCCL ID.
pub fn rendezvous_worker(master_addr: &str, port: u16) -> KernelResult<cudarc::nccl::Id> {
    let addr = format!("{master_addr}:{port}");

    tracing::info!("NCCL rendezvous: connecting to master at {addr}");

    let mut stream = TcpStream::connect(&addr)
        .map_err(|e| KernelError::Other(format!("rendezvous connect to {addr} failed: {e}")))?;

    let mut buf = [0u8; NCCL_ID_BYTES];
    stream
        .read_exact(&mut buf)
        .map_err(|e| KernelError::Other(format!("rendezvous recv failed: {e}")))?;

    // Reconstruct the NCCL ID from raw bytes.
    let internal: [core::ffi::c_char; NCCL_ID_BYTES] = unsafe { std::mem::transmute(buf) };
    let id = cudarc::nccl::Id::uninit(internal);

    tracing::info!("NCCL rendezvous: received ID from master at {addr}");

    Ok(id)
}
