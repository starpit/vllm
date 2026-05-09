// SPDX-License-Identifier: Apache-2.0
//
// Wrapper around Metal 3's `MTLResidencySet` (macOS 15+ / iOS 18+).
// Pins MTL allocations as resident across command buffers so Apple's
// lazy paging doesn't inject non-deterministic stale-page reads when
// the working set is large enough to push the implicit-residency
// tracker over its threshold.
//
// Mirrors `mlx/backend/metal/resident.{h,cpp}` from the MLX repo —
// same API shape (`new`, `insert`, `commit`, `attach_to_queue`),
// adapted to objc-rs. On older macOS the wrapper is a no-op (the
// metal-rs `Device` and `CommandQueue` paths still work; just no
// pinning).
//
// Why this exists for ferrite-metal: `Llama-3.2-3B` (~6 GiB working
// set with weights + KV cache + arena) produced non-deterministic
// decode output even with explicit `enc.use_resource(...)` per
// encoder. Per the MLX source's `CommandEncoder` ctor, the
// production fix is `queue->addResidencySet(set)` so every cmdbuf
// starts with the set wired — no per-encoder residency churn that
// Apple's tracker can race on.

use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, Device};
use objc::runtime::{Object, BOOL, YES};
use objc::{class, msg_send, sel, sel_impl};
use std::sync::{Arc, Mutex};

/// Opaque wrapper around `MTLResidencySet`. `Clone` shares the
/// underlying objc handle via `Arc`, so multiple owners (allocator,
/// worker, queue) can hold references without re-creating the set.
#[derive(Clone)]
pub struct MetalResidencySet {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    /// `MTLResidencySet*` (or null on macOS < 15 / unsupported devices).
    set_ptr: *mut Object,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if !self.set_ptr.is_null() {
            unsafe {
                let _: () = msg_send![self.set_ptr, release];
            }
        }
    }
}

impl MetalResidencySet {
    /// Try to construct a residency set on `device`. Returns a
    /// no-op wrapper on macOS < 15 (where the API doesn't exist) or
    /// when descriptor / set creation fails.
    pub fn new(device: &Device) -> Self {
        let set_ptr = unsafe { try_create_residency_set(device) };
        if !set_ptr.is_null() {
            unsafe {
                let _: () = msg_send![set_ptr, requestResidency];
            }
            if std::env::var_os("FERRITE_METAL_RESIDENCY_DEBUG").is_some() {
                eprintln!("[residency] active (ptr={:?})", set_ptr);
            }
        } else if std::env::var_os("FERRITE_METAL_RESIDENCY_DEBUG").is_some() {
            eprintln!("[residency] inactive (no Metal3 / pre-macOS-15)");
        }
        Self {
            inner: Arc::new(Mutex::new(Inner { set_ptr })),
        }
    }

    /// Returns true on systems where the residency-set API exists and
    /// the set was successfully created. Callers can skip
    /// insert/commit when this is false (the methods are no-ops too,
    /// but skipping the call avoids the lock).
    pub fn is_active(&self) -> bool {
        let inner = self.inner.lock().expect("residency set mutex");
        !inner.set_ptr.is_null()
    }

    /// Add `buffer` to the wired set. Caller is responsible for
    /// calling [`Self::commit`] afterwards (batching multiple inserts
    /// before a single commit cuts down on driver chatter — the MLX
    /// allocator commits per-malloc; we'll commit per-arena which
    /// is roughly per-256MB).
    pub fn insert(&self, buffer: &Buffer) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let buf_ptr = buffer.as_ptr() as *mut Object;
            let _: () = msg_send![inner.set_ptr, addAllocation: buf_ptr];
        }
        if std::env::var_os("FERRITE_METAL_RESIDENCY_DEBUG").is_some() {
            eprintln!(
                "[residency] insert buf={:p} len={}",
                buffer.as_ptr(),
                buffer.length()
            );
        }
    }

    /// Commit pending insertions. Required after a sequence of
    /// `insert` calls before the wired set takes effect.
    pub fn commit(&self) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let _: () = msg_send![inner.set_ptr, commit];
        }
    }

    /// Wire this residency set onto `queue`. Every command buffer
    /// fired through `queue` after this call sees the set's wired
    /// allocations as guaranteed-resident — no per-encoder
    /// `use_resource` declarations needed for those buffers.
    pub fn attach_to_queue(&self, queue: &CommandQueue) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let queue_ptr = queue.as_ptr() as *mut Object;
            let _: () = msg_send![queue_ptr, addResidencySet: inner.set_ptr];
        }
    }
}

/// Build a residency set on `device`. Returns null on macOS < 15 or
/// on any failure (device family check fails, descriptor alloc
/// fails, `newResidencySet:error:` returns nil).
///
/// # Safety
/// Calls Objective-C runtime functions; the returned pointer is
/// retained (caller owns it) and must be released by the wrapper's
/// `Drop`. Passes `&error` as a stack-local nil; we don't surface
/// the error since failure is best-effort fallback.
unsafe fn try_create_residency_set(device: &Device) -> *mut Object {
    let device_ptr = device.as_ptr() as *mut Object;

    // Probe for `newResidencySetWithDescriptor:error:` selector.
    // The selector exists on macOS 15+ runtimes; on older systems
    // `respondsToSelector:` returns NO and we bail out cleanly
    // rather than triggering an unrecognized-selector exception.
    let sel_check = sel!(newResidencySetWithDescriptor:error:);
    let responds: BOOL = msg_send![device_ptr, respondsToSelector: sel_check];
    if responds != YES {
        return std::ptr::null_mut();
    }

    // Build a default `MTLResidencySetDescriptor` (no label, no
    // initialCapacity hints). MLX uses the same path.
    let desc_class = class!(MTLResidencySetDescriptor);
    let desc: *mut Object = msg_send![desc_class, alloc];
    let desc: *mut Object = msg_send![desc, init];
    if desc.is_null() {
        return std::ptr::null_mut();
    }
    let _: () = msg_send![desc, autorelease];

    let mut error: *mut Object = std::ptr::null_mut();
    let set: *mut Object = msg_send![
        device_ptr,
        newResidencySetWithDescriptor: desc
        error: &mut error
    ];
    if set.is_null() {
        return std::ptr::null_mut();
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_does_not_panic() {
        // On macOS 15+ this builds a real set; on older macOS it's
        // a no-op. Either way, no panic.
        let Some(device_info) = crate::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let set = MetalResidencySet::new(&device_info.device);
        // Whether the set is active depends on the host macOS
        // version + GPU family — just exercise the API surface.
        let _ = set.is_active();
        set.commit();
    }

    #[test]
    fn insert_and_attach_roundtrip() {
        let Some(device_info) = crate::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = device_info.device.clone();
        let queue = device.new_command_queue();
        let set = MetalResidencySet::new(&device);

        let buf = device.new_buffer(1024, metal::MTLResourceOptions::StorageModeShared);
        set.insert(&buf);
        set.commit();
        set.attach_to_queue(&queue);
        // The set may or may not be active depending on macOS
        // version — both paths are valid; the assertion is just
        // that none of the calls explode.
    }
}
