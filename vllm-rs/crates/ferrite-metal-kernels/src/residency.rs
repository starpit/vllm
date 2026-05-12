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
// adapted to objc2. On older macOS the wrapper is a no-op (the
// objc2-metal `Device` and `CommandQueue` paths still work; just no
// pinning).

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{class, msg_send, sel};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use std::sync::{Arc, Mutex};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

#[derive(Clone)]
pub struct MetalResidencySet {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    /// `MTLResidencySet*` (or null on macOS < 15 / unsupported devices).
    set_ptr: *mut AnyObject,
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

    pub fn is_active(&self) -> bool {
        let inner = self.inner.lock().expect("residency set mutex");
        !inner.set_ptr.is_null()
    }

    pub fn insert(&self, buffer: &Buffer) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let buf_ptr: *mut AnyObject =
                Retained::as_ptr(buffer) as *const AnyObject as *mut AnyObject;
            let _: () = msg_send![inner.set_ptr, addAllocation: buf_ptr];
        }
        if std::env::var_os("FERRITE_METAL_RESIDENCY_DEBUG").is_some() {
            eprintln!(
                "[residency] insert buf={:p} len={}",
                Retained::as_ptr(buffer),
                buffer.length()
            );
        }
    }

    pub fn commit(&self) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let _: () = msg_send![inner.set_ptr, commit];
        }
    }

    pub fn attach_to_queue(&self, queue: &CommandQueue) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let queue_ptr: *mut AnyObject =
                Retained::as_ptr(queue) as *const AnyObject as *mut AnyObject;
            let _: () = msg_send![queue_ptr, addResidencySet: inner.set_ptr];
        }
    }

    /// Phase-A MTL4 helper: invoke `useResidencySet:` on a
    /// `MTL4CommandBuffer`. MTL4 cmdbufs declare residency per-buffer
    /// (unlike MTL3 cmdbufs which inherit from the queue-attached
    /// set), so each fresh `MTL4CommandBuffer` needs this call between
    /// `beginCommandBufferWithAllocator` and `endCommandBuffer`.
    /// Takes a raw `*mut AnyObject` rather than a typed reference so
    /// the kernels crate can stay free of an `objc2-metal` MTL4 dep.
    ///
    /// # Safety
    /// `cb_ptr` must be a non-null pointer to a live MTL4CommandBuffer
    /// in the recording state.
    pub unsafe fn attach_to_mtl4_command_buffer(&self, cb_ptr: *mut AnyObject) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() || cb_ptr.is_null() {
            return;
        }
        let _: () = msg_send![cb_ptr, useResidencySet: inner.set_ptr];
    }

    /// Belt-and-braces MTL4 helper: attach to the MTL4 command queue
    /// via `addResidencySet:`. The per-cmdbuf `useResidencySet:` is
    /// documented as sufficient, but on some hardware (M1 Max
    /// observed) additionally attaching to the queue is required for
    /// indirectly-addressed buffers (paged KV cache) to stay
    /// resident across cmdbufs. Idempotent — Metal dedupes.
    ///
    /// # Safety
    /// `queue_ptr` must be a non-null pointer to a live MTL4-shaped
    /// command queue.
    pub unsafe fn attach_to_mtl4_queue(&self, queue_ptr: *mut AnyObject) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() || queue_ptr.is_null() {
            return;
        }
        let _: () = msg_send![queue_ptr, addResidencySet: inner.set_ptr];
    }
}

/// Build a residency set on `device`. Returns null on macOS < 15 or
/// on any failure (device family check fails, descriptor alloc
/// fails, `newResidencySet:error:` returns nil).
unsafe fn try_create_residency_set(device: &Device) -> *mut AnyObject {
    let device_ptr: *mut AnyObject = Retained::as_ptr(device) as *const AnyObject as *mut AnyObject;

    // Probe for `newResidencySetWithDescriptor:error:` selector. The
    // selector exists on macOS 15+ runtimes; on older systems
    // `respondsToSelector:` returns NO and we bail out cleanly rather
    // than triggering an unrecognized-selector exception.
    let sel_check = sel!(newResidencySetWithDescriptor:error:);
    let responds: Bool = msg_send![device_ptr, respondsToSelector: sel_check];
    if !responds.as_bool() {
        return std::ptr::null_mut();
    }

    // Build a default `MTLResidencySetDescriptor` (no label, no
    // initialCapacity hints). MLX uses the same path.
    let desc_class = class!(MTLResidencySetDescriptor);
    let desc: *mut AnyObject = msg_send![desc_class, alloc];
    let desc: *mut AnyObject = msg_send![desc, init];
    if desc.is_null() {
        return std::ptr::null_mut();
    }
    let _: *mut AnyObject = msg_send![desc, autorelease];

    let mut error: *mut AnyObject = std::ptr::null_mut();
    let set: *mut AnyObject = msg_send![
        device_ptr,
        newResidencySetWithDescriptor: desc,
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
    use objc2_metal::MTLResourceOptions;

    #[test]
    fn construction_does_not_panic() {
        let Some(device_info) = crate::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let set = MetalResidencySet::new(&device_info.device);
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
        let queue = device.newCommandQueue().expect("newCommandQueue");
        let set = MetalResidencySet::new(&device);

        let buf = device
            .newBufferWithLength_options(1024, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength");
        set.insert(&buf);
        set.commit();
        set.attach_to_queue(&queue);
    }
}
