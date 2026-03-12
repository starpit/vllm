// SPDX-License-Identifier: Apache-2.0
//! TCP-based store for inter-process NCCL initialization.
//!
//! When using an external launcher (torchrun, mpirun, SLURM), each process
//! needs to exchange the NCCL unique ID and coordinate memory allocation.
//! This module provides a simple synchronous TCP store for that purpose.
//!
//! Rank 0 acts as the server:
//! - Generates the NCCL unique ID
//! - Listens on `MASTER_ADDR:MASTER_PORT`
//! - Sends the 128-byte ID to each connecting rank
//!
//! Ranks 1..N-1 connect to rank 0 and receive the ID.
//!
//! Also provides an all-reduce MIN for memory coordination: each rank sends
//! its available memory to rank 0, which computes the minimum and broadcasts
//! it back.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use anyhow::{Context, Result};
use tracing::info;

/// Size of an NCCL unique ID in bytes.
#[cfg(feature = "nccl")]
const NCCL_ID_BYTES: usize = 128;

/// Exchange the NCCL unique ID across processes via TCP.
///
/// - Rank 0: generates a new NCCL ID, accepts `world_size - 1` connections,
///   and sends the ID to each.
/// - Rank N > 0: connects to `master_addr:master_port` and receives the ID.
///
/// Returns the raw 128-byte NCCL unique ID (same value on all ranks).
#[cfg(feature = "nccl")]
pub fn exchange_nccl_id(
    rank: usize,
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    if rank == 0 {
        exchange_nccl_id_rank0(world_size, master_addr, master_port)
    } else {
        exchange_nccl_id_worker(master_addr, master_port)
    }
}

/// Rank 0: generate NCCL ID and distribute to all other ranks.
#[cfg(feature = "nccl")]
fn exchange_nccl_id_rank0(
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    // Generate the NCCL unique ID.
    let nccl_id = crate::NcclId::new().context("failed to generate NCCL unique ID")?;
    let id_bytes = *nccl_id.raw();

    if world_size == 1 {
        return Ok(id_bytes);
    }

    let bind_addr = format!("{master_addr}:{master_port}");
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("rank 0: failed to bind to {bind_addr}"))?;
    info!(
        "Rank 0: listening on {} for {} worker(s)",
        bind_addr,
        world_size - 1
    );

    // Accept connections from ranks 1..world_size-1.
    // Cast c_char bytes to u8 for network I/O.
    let id_as_u8: &[u8; NCCL_ID_BYTES] = unsafe { &*(&id_bytes as *const _ as *const _) };
    for i in 1..world_size {
        let (mut stream, peer_addr) = listener
            .accept()
            .with_context(|| format!("rank 0: failed to accept connection {i}"))?;
        info!(
            "Rank 0: accepted connection from {} (rank {})",
            peer_addr, i
        );
        stream
            .write_all(id_as_u8)
            .with_context(|| format!("rank 0: failed to send NCCL ID to rank {i}"))?;
    }

    Ok(id_bytes)
}

/// Rank N > 0: connect to rank 0 and receive the NCCL unique ID.
#[cfg(feature = "nccl")]
fn exchange_nccl_id_worker(
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    let addr = format!("{master_addr}:{master_port}");

    // Retry connection with backoff — rank 0 may not be listening yet.
    let mut stream = None;
    for attempt in 0..30 {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt < 29 {
                    let delay = std::time::Duration::from_millis(500);
                    info!(
                        "Worker: connection to {} failed (attempt {}): {}, retrying in {:?}",
                        addr,
                        attempt + 1,
                        e,
                        delay
                    );
                    std::thread::sleep(delay);
                } else {
                    return Err(e).with_context(|| {
                        format!("worker: failed to connect to {addr} after 30 attempts")
                    });
                }
            }
        }
    }
    let mut stream = stream.unwrap();

    let mut buf = [0u8; NCCL_ID_BYTES];
    stream
        .read_exact(&mut buf)
        .context("worker: failed to read NCCL ID from rank 0")?;

    // Reinterpret u8 bytes as c_char.
    let id_bytes: [core::ffi::c_char; NCCL_ID_BYTES] = unsafe { std::mem::transmute(buf) };
    Ok(id_bytes)
}

/// All-reduce MIN of a `usize` value across all ranks via TCP.
///
/// - Rank 0: accepts `world_size - 1` connections, reads each rank's value,
///   computes the minimum (including its own), and sends the result back.
/// - Rank N > 0: connects to rank 0, sends its value, receives the minimum.
///
/// Returns the minimum value across all ranks.
pub fn allreduce_min(
    rank: usize,
    world_size: usize,
    value: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<usize> {
    if world_size == 1 {
        return Ok(value);
    }

    // Use master_port + 1 to avoid conflicting with the NCCL ID exchange.
    let port = master_port + 1;

    if rank == 0 {
        allreduce_min_rank0(world_size, value, master_addr, port)
    } else {
        allreduce_min_worker(value, master_addr, port)
    }
}

/// Rank 0: collect values, compute min, broadcast result.
fn allreduce_min_rank0(
    world_size: usize,
    local_value: usize,
    master_addr: &str,
    port: u16,
) -> Result<usize> {
    let bind_addr = format!("{master_addr}:{port}");
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("rank 0: failed to bind to {bind_addr} for allreduce"))?;

    let mut min_value = local_value;
    let mut streams = Vec::with_capacity(world_size - 1);

    for i in 1..world_size {
        let (mut stream, _) = listener
            .accept()
            .with_context(|| format!("rank 0: failed to accept allreduce connection {i}"))?;

        let mut buf = [0u8; 8];
        stream
            .read_exact(&mut buf)
            .with_context(|| format!("rank 0: failed to read value from rank {i}"))?;
        let remote_value = usize::from_le_bytes(buf);
        min_value = min_value.min(remote_value);
        streams.push(stream);
    }

    info!(
        "Rank 0: allreduce MIN = {} bytes ({:.1} GB)",
        min_value,
        min_value as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Broadcast the minimum back to all workers.
    let result_bytes = min_value.to_le_bytes();
    for (i, mut stream) in streams.into_iter().enumerate() {
        stream
            .write_all(&result_bytes)
            .with_context(|| format!("rank 0: failed to send min to rank {}", i + 1))?;
    }

    Ok(min_value)
}

/// Rank N > 0: send value to rank 0, receive the minimum.
fn allreduce_min_worker(value: usize, master_addr: &str, port: u16) -> Result<usize> {
    let addr = format!("{master_addr}:{port}");

    // Retry connection with backoff.
    let mut stream = None;
    for attempt in 0..30 {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt < 29 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                } else {
                    return Err(e).with_context(|| {
                        format!("worker: failed to connect to {addr} for allreduce")
                    });
                }
            }
        }
    }
    let mut stream = stream.unwrap();

    // Send our value.
    stream
        .write_all(&value.to_le_bytes())
        .context("worker: failed to send value for allreduce")?;

    // Receive the minimum.
    let mut buf = [0u8; 8];
    stream
        .read_exact(&mut buf)
        .context("worker: failed to read allreduce result")?;

    Ok(usize::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allreduce_min_single_rank() {
        let result = allreduce_min(0, 1, 42, "127.0.0.1", 0).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_allreduce_min_multi_rank() {
        let world_size = 4;
        let values = [100usize, 50, 200, 25];
        let expected_min = 25;

        // Use port 0 to let the OS assign a free port — but we need to know it.
        // Instead, pick a high port that's unlikely to conflict.
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19876);

        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let value = values[rank];
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();

        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, expected_min);
        }
    }

    #[test]
    fn test_allreduce_min_two_ranks() {
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19878);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 { 1000 } else { 500 };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 500);
        }
    }

    #[test]
    fn test_allreduce_min_identical_values() {
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19879);
        let world_size = 3;
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, 42, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 42);
        }
    }

    #[test]
    fn test_allreduce_min_rank0_has_minimum() {
        // Ensure rank 0's own value is included in the min computation.
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19880);
        let world_size = 3;
        // Rank 0 has the smallest value.
        let values = [1usize, 100, 200];
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let value = values[rank];
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 1);
        }
    }

    #[test]
    fn test_allreduce_min_large_world_size() {
        let world_size = 8;
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19881);
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                // Rank 5 has the minimum.
                let value = if rank == 5 { 7 } else { 1000 + rank };
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 7);
        }
    }

    #[test]
    fn test_allreduce_min_zero_value() {
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19882);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 { 0 } else { 100 };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 0);
        }
    }

    #[test]
    fn test_allreduce_min_max_usize() {
        // Edge case: usize::MAX should be representable.
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19883);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 {
                    usize::MAX
                } else {
                    usize::MAX - 1
                };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), usize::MAX - 1);
        }
    }

    #[test]
    fn test_allreduce_sequential_calls() {
        // Two sequential allreduce rounds on different ports (simulating
        // NCCL ID exchange then memory coordination).
        let port1: u16 = portpicker::pick_unused_port().unwrap_or(19884);
        let port2: u16 = portpicker::pick_unused_port().unwrap_or(19885);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let v1 = allreduce_min(rank, 2, 100 + rank, "127.0.0.1", port1).unwrap();
                    let v2 = allreduce_min(rank, 2, 200 + rank, "127.0.0.1", port2).unwrap();
                    (v1, v2)
                })
            })
            .collect();
        for handle in handles {
            let (v1, v2) = handle.join().unwrap();
            assert_eq!(v1, 100); // min(100, 101)
            assert_eq!(v2, 200); // min(200, 201)
        }
    }

    #[test]
    #[cfg(feature = "nccl")]
    fn test_nccl_id_exchange_simulated() {
        // We can't call real NCCL ID generation without CUDA, so test the TCP
        // plumbing with a mock: rank 0 serves fixed bytes, workers receive them.
        let world_size = 3;
        let port: u16 = portpicker::pick_unused_port().unwrap_or(19877);

        // Simulate by directly testing the TCP layer with raw bytes.
        let test_bytes = [42i8; NCCL_ID_BYTES];

        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                std::thread::spawn(move || -> [u8; NCCL_ID_BYTES] {
                    if rank == 0 {
                        // Server: send test_bytes to each worker.
                        let bind_addr = format!("127.0.0.1:{port}");
                        let listener = TcpListener::bind(&bind_addr).unwrap();
                        let id_as_u8: &[u8; NCCL_ID_BYTES] =
                            unsafe { &*(&test_bytes as *const _ as *const _) };
                        for _ in 1..world_size {
                            let (mut stream, _) = listener.accept().unwrap();
                            stream.write_all(id_as_u8).unwrap();
                        }
                        unsafe { std::mem::transmute(test_bytes) }
                    } else {
                        // Worker: connect and receive.
                        let addr = format!("127.0.0.1:{port}");
                        let mut stream = loop {
                            match TcpStream::connect(&addr) {
                                Ok(s) => break s,
                                Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
                            }
                        };
                        let mut buf = [0u8; NCCL_ID_BYTES];
                        stream.read_exact(&mut buf).unwrap();
                        buf
                    }
                })
            })
            .collect();

        let expected: [u8; NCCL_ID_BYTES] = unsafe { std::mem::transmute([42i8; NCCL_ID_BYTES]) };
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, expected);
        }
    }
}
