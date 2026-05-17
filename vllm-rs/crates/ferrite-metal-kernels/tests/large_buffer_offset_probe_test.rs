// SPDX-License-Identifier: Apache-2.0
//! Probes whether Apple's `Shared` MTLBuffer correctly exposes bytes
//! at offsets > 4 GiB to GPU kernels. Empirically the production code
//! sees a `Qwen3-MoE-30B-A3B-4bit` `switch_mlp.down_proj.weight` bound
//! at offset 0x12ad13db0 (4.67 GiB) inside a 5.16 GiB shard buffer
//! return zeros from `MTLBuffer.contents()`; the kernel that consumes
//! it produces outputs consistent with `W == 0`.
//!
//! This standalone test removes the loader/dispatch complexity:
//!   1. Allocate a `Shared` MTLBuffer ≥ 5 GiB on the same device.
//!   2. Write a distinctive 32-byte pattern at three offsets:
//!      1 GiB, 4 GiB, 4.7 GiB.
//!   3. Use a GPU blit to copy 32 bytes from each of those offsets
//!      into a small (4 KiB) Shared destination buffer.
//!   4. CPU-read the destination buffer and check whether the bytes
//!      survived. If the 4.7 GiB sample reads back as zeros, Apple's
//!      Shared buffer addressing capped at 4 GiB — confirms the
//!      production hypothesis and motivates splitting register_mmap
//!      shards.
//!
//! `#[ignore]`d so it doesn't run in the default test sweep — the
//! 5 GiB allocation is too aggressive for shared CI. Run explicitly
//! when reproducing:
//!   cargo test --release -p ferrite-metal-kernels \
//!       --test large_buffer_offset_probe_test \
//!       -- --ignored --nocapture

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLResourceOptions,
};

const GIB: usize = 1024 * 1024 * 1024;

#[test]
#[ignore]
fn shared_buffer_above_4_gib_offset_round_trip() {
    let mdev = detect_device().expect("detect_device");
    let device = &mdev.device;
    let queue = device.newCommandQueue().expect("newCommandQueue");

    let max_len = device.maxBufferLength() as usize;
    eprintln!("device.maxBufferLength = {} bytes ({:.2} GiB)", max_len, max_len as f64 / GIB as f64);

    let big_len: usize = 5 * GIB + 256 * 1024 * 1024; // 5.25 GiB
    if max_len < big_len {
        panic!(
            "device.maxBufferLength = {} < required {} — cannot run >4 GiB probe on this device",
            max_len, big_len
        );
    }

    // 5.25 GiB Shared buffer.
    let big = device
        .newBufferWithLength_options(big_len, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength(big) returned nil");
    eprintln!(
        "allocated big buffer: length={} contents()={:p}",
        big.length(),
        big.contents().as_ptr()
    );

    // 4 KiB Shared destination for GPU blit-back-to-CPU.
    let small = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength(small) returned nil");

    // CPU-write a 32-byte pattern at three offsets.
    let offsets = [1 * GIB, 4 * GIB, 4 * GIB + 700 * 1024 * 1024];
    let patterns: Vec<[u8; 32]> = offsets
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut p = [0u8; 32];
            for (j, b) in p.iter_mut().enumerate() {
                *b = ((i + 1) as u8).wrapping_mul((j as u8).wrapping_add(1));
            }
            p
        })
        .collect();

    // Write via CPU contents() to test CPU-mapping at those offsets.
    let cpu_base = big.contents().as_ptr() as *mut u8;
    for (i, &off) in offsets.iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(patterns[i].as_ptr(), cpu_base.add(off), 32);
        }
    }

    // Read back via CPU to see if the writes stuck.
    eprintln!("--- CPU readback at each offset ---");
    for (i, &off) in offsets.iter().enumerate() {
        let mut buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_base.add(off), buf.as_mut_ptr(), 32);
        }
        let matches = buf == patterns[i];
        eprintln!(
            "  offset=0x{:x} ({:.2} GiB) cpu_readback_matches_pattern={} first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &buf[..8]
        );
    }

    // GPU blit each offset to a separate slot in `small`.
    let cmdbuf = queue.commandBuffer().expect("commandBuffer");
    let blit = cmdbuf.blitCommandEncoder().expect("blitCommandEncoder");
    for (i, &off) in offsets.iter().enumerate() {
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &big,
                off,
                &small,
                i * 64,
                32,
            );
        }
    }
    blit.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    // CPU-read the small buffer (which is < 4 KiB, definitely within
    // any mapping limit) to see what the GPU saw.
    let small_base = small.contents().as_ptr() as *const u8;
    eprintln!("--- GPU-blit readback (via small buffer) at each offset ---");
    let mut any_mismatch = false;
    for (i, &off) in offsets.iter().enumerate() {
        let mut buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(small_base.add(i * 64), buf.as_mut_ptr(), 32);
        }
        let matches = buf == patterns[i];
        if !matches {
            any_mismatch = true;
        }
        eprintln!(
            "  offset=0x{:x} ({:.2} GiB) gpu_blit_matches_pattern={} first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &buf[..8]
        );
    }
    assert!(
        !any_mismatch,
        "GPU blit failed to read back the CPU-written pattern at one or more offsets — \
         this confirms Apple Silicon `Shared` MTLBuffer addressing breaks above some limit \
         (likely 4 GiB)."
    );

    // Suppress unused-import warnings.
    let _ = NonNull::<c_void>::dangling();
}

/// Matches production `register_mmap` shape: source buffer is a
/// `newBufferWithBytesNoCopy` wrapper around a real mmap, destination
/// is a freshly-allocated `Shared` buffer, blit copies the entire
/// mmap from src offset 0 to dst offset `shift`. Then probes the
/// destination at several offsets (including > 4 GiB).
///
/// Uses a 6 GiB file from `/tmp`; if the file doesn't exist the test
/// creates one filled with a deterministic pattern.
#[test]
#[ignore]
fn nocopy_source_blit_above_4_gib_round_trip() {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    let mdev = detect_device().expect("detect_device");
    let device = &mdev.device;
    let queue = device.newCommandQueue().expect("newCommandQueue");

    let path = "/tmp/ferrite_large_buffer_probe.bin";
    let len: usize = 5 * GIB + 256 * 1024 * 1024; // 5.25 GiB

    // Create the file once with a deterministic pattern if it doesn't
    // exist (or is the wrong size).
    let needs_create = match std::fs::metadata(path) {
        Ok(m) => m.len() as usize != len,
        Err(_) => true,
    };
    if needs_create {
        eprintln!("creating {path} ({} bytes, this may take a moment)", len);
        let mut f = OpenOptions::new()
            .create(true).write(true).truncate(true).open(path).unwrap();
        let chunk: Vec<u8> = (0..(1024 * 1024)).map(|i| (i as u8) ^ ((i >> 8) as u8)).collect();
        for off in (0..len).step_by(chunk.len()) {
            let remain = (len - off).min(chunk.len());
            f.write_all(&chunk[..remain]).unwrap();
        }
        f.flush().unwrap();
    }

    // mmap the file via raw mmap (avoid pulling in memmap2 / libc as
    // dev-deps for this one probe).
    let file = std::fs::File::open(path).unwrap();
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    let mmap_len = std::fs::metadata(path).unwrap().len() as usize;
    let prot_read: i32 = 0x01;
    let map_private: i32 = 0x0002;
    let mmap_addr = unsafe {
        let addr = mmap_libc(std::ptr::null_mut(), mmap_len, prot_read, map_private, fd, 0);
        assert!(addr as isize != -1, "mmap failed");
        addr as *const u8
    };
    eprintln!("mmap base={:p} len={}", mmap_addr, mmap_len);

    // Build a NoCopy source MTLBuffer wrapping the mmap (exactly like
    // `register_mmap`).
    let page_size: usize = 16384;
    let src_buffer_len = (mmap_len + page_size - 1) & !(page_size - 1);
    let src_buffer = unsafe {
        let bytes = NonNull::new(mmap_addr as *mut c_void).expect("non-null");
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                bytes,
                src_buffer_len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .expect("newBufferWithBytesNoCopy returned nil")
    };
    eprintln!("src_buffer length={}", src_buffer.length());

    // Destination: fresh Shared MTLBuffer same length.
    let shift: usize = 10; // simulate non-zero shift like register_mmap
    let dst_capacity = mmap_len + shift;
    let dst = device
        .newBufferWithLength_options(dst_capacity, MTLResourceOptions::StorageModeShared)
        .expect("dst alloc");
    eprintln!("dst length={} contents()={:p}", dst.length(), dst.contents().as_ptr());

    // Blit src[0..len] → dst[shift..shift+len], CHUNKED to ≤2 GiB
    // per call to dodge Apple's silent ~4 GiB blit-size cap on
    // NoCopy-backed sources.
    let cmdbuf = queue.commandBuffer().unwrap();
    let blit = cmdbuf.blitCommandEncoder().unwrap();
    const CHUNK: usize = 2 * GIB;
    let mut copied = 0usize;
    while copied < mmap_len {
        let n = (mmap_len - copied).min(CHUNK);
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buffer, copied, &dst, shift + copied, n,
            );
        }
        copied += n;
    }
    blit.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    let status = cmdbuf.status();
    let err = cmdbuf.error().map(|e| format!("{:?}", e)).unwrap_or_else(|| "(no error)".into());
    eprintln!("blit cmdbuf status={:?} error={}", status, err);

    // Sample several offsets and compare CPU readbacks of dst to the
    // mmap. Includes 4.68 GiB which is where production's down_proj
    // binding lands.
    let mut file2 = std::fs::File::open(path).unwrap();
    let mmap_offsets = [
        1 * GIB,
        4 * GIB,
        4 * GIB + 700 * 1024 * 1024, // ~4.68 GiB
        5 * GIB,
    ];
    let dst_base = dst.contents().as_ptr() as *const u8;
    eprintln!("--- CPU readback of dst after blit ---");
    let mut any_mismatch = false;
    for off in mmap_offsets {
        let mut file_buf = [0u8; 32];
        file2.seek(SeekFrom::Start(off as u64)).unwrap();
        file2.read_exact(&mut file_buf).unwrap();

        let mut dst_buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(dst_base.add(off + shift), dst_buf.as_mut_ptr(), 32);
        }
        let matches = file_buf == dst_buf;
        if !matches { any_mismatch = true; }
        eprintln!(
            "  mmap_off=0x{:x} ({:.2} GiB) cpu_dst_matches_file={} file_first8={:02x?} dst_first8={:02x?}",
            off, off as f64 / GIB as f64, matches, &file_buf[..8], &dst_buf[..8]
        );
    }

    // Also GPU blit dst → small probe buffer to see what GPU sees.
    let small = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("small");
    let cmdbuf2 = queue.commandBuffer().unwrap();
    let blit2 = cmdbuf2.blitCommandEncoder().unwrap();
    for (i, &off) in mmap_offsets.iter().enumerate() {
        unsafe {
            blit2.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &dst, off + shift, &small, i * 64, 32,
            );
        }
    }
    blit2.endEncoding();
    cmdbuf2.commit();
    cmdbuf2.waitUntilCompleted();
    let small_base = small.contents().as_ptr() as *const u8;
    eprintln!("--- GPU-blit readback of dst ---");
    for (i, &off) in mmap_offsets.iter().enumerate() {
        let mut file_buf = [0u8; 32];
        file2.seek(SeekFrom::Start(off as u64)).unwrap();
        file2.read_exact(&mut file_buf).unwrap();

        let mut small_buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(small_base.add(i * 64), small_buf.as_mut_ptr(), 32);
        }
        let matches = file_buf == small_buf;
        if !matches { any_mismatch = true; }
        eprintln!(
            "  mmap_off=0x{:x} ({:.2} GiB) gpu_dst_matches_file={} file_first8={:02x?} gpu_first8={:02x?}",
            off, off as f64 / GIB as f64, matches, &file_buf[..8], &small_buf[..8]
        );
    }

    assert!(!any_mismatch, "blit-from-NoCopy round trip failed at some offset");
}

// Raw mmap binding (avoids pulling in libc/memmap2 as dev-deps).
extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
}
unsafe fn mmap_libc(
    addr: *mut std::ffi::c_void,
    length: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    offset: i64,
) -> *mut std::ffi::c_void {
    mmap(addr, length, prot, flags, fd, offset)
}
