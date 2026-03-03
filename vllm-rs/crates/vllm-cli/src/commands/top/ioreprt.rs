// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! IOReport FFI bindings for Apple Silicon GPU/power metrics.
//!
//! Adapted from vladkens/macmon (MIT).

#![allow(non_snake_case, clippy::missing_safety_doc)]

use std::marker::{PhantomData, PhantomPinned};
use std::ptr::null;

use core_foundation::base::{
    CFAllocatorRef, CFRelease, CFTypeRef, kCFAllocatorDefault, kCFAllocatorNull,
};
use core_foundation::dictionary::{
    CFDictionaryCreateMutableCopy, CFDictionaryGetCount, CFDictionaryGetValue, CFDictionaryRef,
    CFMutableDictionaryRef,
};
use core_foundation::string::{
    CFStringCreateWithBytesNoCopy, CFStringGetCString, CFStringRef, kCFStringEncodingUTF8,
};

type CVoidRef = *const std::ffi::c_void;

// ---------------------------------------------------------------------------
// IOReport FFI declarations
// ---------------------------------------------------------------------------

#[repr(C)]
struct IOReportSubscription {
    _data: [u8; 0],
    _phantom: PhantomData<(*mut u8, PhantomPinned)>,
}
type IOReportSubscriptionRef = *const IOReportSubscription;

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        a: CFStringRef,
        b: CFStringRef,
        c: u64,
        d: u64,
        e: u64,
    ) -> CFDictionaryRef;
    fn IOReportMergeChannels(a: CFDictionaryRef, b: CFDictionaryRef, nil: CFTypeRef);
    fn IOReportCreateSubscription(
        a: CVoidRef,
        b: CFMutableDictionaryRef,
        c: *mut CFMutableDictionaryRef,
        d: u64,
        e: CFTypeRef,
    ) -> IOReportSubscriptionRef;
    fn IOReportCreateSamples(
        a: IOReportSubscriptionRef,
        b: CFMutableDictionaryRef,
        c: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportCreateSamplesDelta(
        a: CFDictionaryRef,
        b: CFDictionaryRef,
        c: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportChannelGetGroup(a: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetChannelName(a: CFDictionaryRef) -> CFStringRef;
    fn IOReportSimpleGetIntegerValue(a: CFDictionaryRef, b: i32) -> i64;
    fn IOReportChannelGetUnitLabel(a: CFDictionaryRef) -> CFStringRef;
    fn IOReportStateGetCount(a: CFDictionaryRef) -> i32;
    fn IOReportStateGetNameForIndex(a: CFDictionaryRef, b: i32) -> CFStringRef;
    fn IOReportStateGetResidency(a: CFDictionaryRef, b: i32) -> i64;
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const i8) -> CFMutableDictionaryRef;
    fn IOServiceGetMatchingServices(
        mainPort: u32,
        matching: CFDictionaryRef,
        existing: *mut u32,
    ) -> i32;
    fn IOIteratorNext(iterator: u32) -> u32;
    fn IORegistryEntryGetName(entry: u32, name: *mut i8) -> i32;
    fn IORegistryEntryCreateCFProperties(
        entry: u32,
        properties: *mut CFMutableDictionaryRef,
        allocator: CFAllocatorRef,
        options: u32,
    ) -> i32;
    fn IOObjectRelease(obj: u32) -> u32;
}

use core_foundation::array::{CFArrayGetCount, CFArrayGetValueAtIndex, CFArrayRef};
use core_foundation::data::{CFDataGetBytes, CFDataGetLength, CFDataRef};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cfstr(val: &str) -> CFStringRef {
    unsafe {
        CFStringCreateWithBytesNoCopy(
            kCFAllocatorDefault,
            val.as_ptr(),
            val.len() as isize,
            kCFStringEncodingUTF8,
            0u8,
            kCFAllocatorNull,
        )
    }
}

fn from_cfstr(val: CFStringRef) -> String {
    if val.is_null() {
        return String::new();
    }
    unsafe {
        let mut buf = vec![0i8; 256];
        CFStringGetCString(val, buf.as_mut_ptr(), 256, kCFStringEncodingUTF8);
        std::ffi::CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .to_string()
    }
}

fn cfdict_get_val(dict: CFDictionaryRef, key: &str) -> Option<CFTypeRef> {
    if dict.is_null() {
        return None;
    }
    let k = cfstr(key);
    let val = unsafe { CFDictionaryGetValue(dict, k as CFTypeRef) };
    if val.is_null() { None } else { Some(val) }
}

fn cfio_get_residencies(item: CFDictionaryRef) -> Vec<(String, i64)> {
    let count = unsafe { IOReportStateGetCount(item) };
    let mut res = Vec::with_capacity(count as usize);
    for i in 0..count {
        let name = unsafe { from_cfstr(IOReportStateGetNameForIndex(item, i)) };
        let val = unsafe { IOReportStateGetResidency(item, i) };
        res.push((name, val));
    }
    res
}

// ---------------------------------------------------------------------------
// SoC info: GPU frequency table from IORegistry
// ---------------------------------------------------------------------------

fn get_gpu_freqs() -> Vec<u32> {
    unsafe {
        let matching = IOServiceMatching(c"AppleARMIODevice".as_ptr());
        if matching.is_null() {
            return Vec::new();
        }
        let mut iter: u32 = 0;
        if IOServiceGetMatchingServices(0, matching, &mut iter) != 0 {
            return Vec::new();
        }
        let mut freqs = Vec::new();
        loop {
            let entry = IOIteratorNext(iter);
            if entry == 0 {
                break;
            }
            let mut name_buf = [0i8; 128];
            if IORegistryEntryGetName(entry, name_buf.as_mut_ptr()) == 0 {
                let name = std::ffi::CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .to_string();
                if name == "pmgr" {
                    let mut props: CFMutableDictionaryRef = std::ptr::null_mut();
                    if IORegistryEntryCreateCFProperties(entry, &mut props, kCFAllocatorDefault, 0)
                        == 0
                        && !props.is_null()
                    {
                        freqs = read_dvfs_freqs(props as CFDictionaryRef, "voltage-states9");
                        CFRelease(props as CFTypeRef);
                    }
                }
            }
            IOObjectRelease(entry);
        }
        IOObjectRelease(iter);
        freqs
    }
}

fn read_dvfs_freqs(dict: CFDictionaryRef, key: &str) -> Vec<u32> {
    let Some(val) = cfdict_get_val(dict, key) else {
        return Vec::new();
    };
    let data = val as CFDataRef;
    let len = unsafe { CFDataGetLength(data) } as usize;
    if len == 0 || !len.is_multiple_of(8) {
        return Vec::new();
    }
    let mut buf = vec![0u8; len];
    unsafe {
        CFDataGetBytes(
            data,
            core_foundation::base::CFRange {
                location: 0,
                length: len as isize,
            },
            buf.as_mut_ptr(),
        );
    }
    // Each 8-byte pair: [freq_u32_le, voltage_u32_le]
    buf.chunks_exact(8)
        .map(|chunk| {
            let freq = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            freq / 1_000_000 // Hz → MHz
        })
        .filter(|f| *f > 0)
        .collect()
}

// ---------------------------------------------------------------------------
// Public: IOReport sampler
// ---------------------------------------------------------------------------

pub struct IOReportSampler {
    subscription: IOReportSubscriptionRef,
    channels: CFMutableDictionaryRef,
    prev_sample: Option<CFDictionaryRef>,
    gpu_freqs: Vec<u32>,
}

// IOReport handles are thread-safe in practice (single-thread sampler).
unsafe impl Send for IOReportSampler {}

impl IOReportSampler {
    pub fn new() -> Option<Self> {
        let gpu_freqs = get_gpu_freqs();

        // Subscribe to GPU Stats + Energy Model channels.
        let gpu_chan = unsafe {
            IOReportCopyChannelsInGroup(
                cfstr("GPU Stats"),
                cfstr("GPU Performance States"),
                0,
                0,
                0,
            )
        };
        if gpu_chan.is_null() {
            return None;
        }

        let energy_chan = unsafe {
            IOReportCopyChannelsInGroup(cfstr("Energy Model"), std::ptr::null(), 0, 0, 0)
        };

        // Merge channels.
        let merged = unsafe {
            CFDictionaryCreateMutableCopy(
                kCFAllocatorDefault,
                CFDictionaryGetCount(gpu_chan),
                gpu_chan,
            )
        };
        if !energy_chan.is_null() {
            unsafe {
                IOReportMergeChannels(merged as CFDictionaryRef, energy_chan, null());
            }
        }

        // Create subscription.
        let mut sub_out: CFMutableDictionaryRef = std::ptr::null_mut();
        let subs = unsafe { IOReportCreateSubscription(null(), merged, &mut sub_out, 0, null()) };
        if subs.is_null() {
            return None;
        }

        Some(Self {
            subscription: subs,
            channels: merged,
            prev_sample: None,
            gpu_freqs,
        })
    }

    /// Take a sample and compute delta from previous sample.
    /// Returns (gpu_util_pct, gpu_freq_mhz, gpu_power_w).
    pub fn sample(&mut self) -> (f64, f64, f64) {
        let sample = unsafe { IOReportCreateSamples(self.subscription, self.channels, null()) };
        if sample.is_null() {
            return (0.0, 0.0, 0.0);
        }

        let result = if let Some(prev) = self.prev_sample {
            let delta = unsafe { IOReportCreateSamplesDelta(prev, sample, null()) };
            unsafe { CFRelease(prev as CFTypeRef) };
            let r = self.parse_delta(delta);
            if !delta.is_null() {
                unsafe { CFRelease(delta as CFTypeRef) };
            }
            r
        } else {
            (0.0, 0.0, 0.0)
        };

        self.prev_sample = Some(sample);
        result
    }

    fn parse_delta(&self, delta: CFDictionaryRef) -> (f64, f64, f64) {
        if delta.is_null() {
            return (0.0, 0.0, 0.0);
        }
        let Some(items) = cfdict_get_val(delta, "IOReportChannels") else {
            return (0.0, 0.0, 0.0);
        };
        let items = items as CFArrayRef;
        let count = unsafe { CFArrayGetCount(items) };

        let mut gpu_util = 0.0f64;
        let mut gpu_freq = 0.0f64;
        let mut gpu_power = 0.0f64;

        for i in 0..count {
            let item = unsafe { CFArrayGetValueAtIndex(items, i) } as CFDictionaryRef;
            let group = unsafe { from_cfstr(IOReportChannelGetGroup(item)) };
            let channel = unsafe { from_cfstr(IOReportChannelGetChannelName(item)) };

            if group == "GPU Stats" && channel == "GPUPH" {
                let (freq, util) = self.calc_gpu_freq_util(item);
                gpu_freq = freq;
                gpu_util = util * 100.0; // 0-1 → 0-100
            } else if group == "Energy Model" && channel == "GPU Energy" {
                let unit = unsafe { from_cfstr(IOReportChannelGetUnitLabel(item)) };
                let val = unsafe { IOReportSimpleGetIntegerValue(item, 0) } as f64;
                // Convert energy to watts (samples are ~1s apart).
                // The delta is energy over the sample interval.
                gpu_power = energy_to_watts(val, &unit);
            }
        }

        (gpu_util, gpu_freq, gpu_power)
    }

    fn calc_gpu_freq_util(&self, item: CFDictionaryRef) -> (f64, f64) {
        let residencies = cfio_get_residencies(item);
        if residencies.is_empty() {
            return (0.0, 0.0);
        }

        // Find first active state (skip "OFF").
        let offset = residencies
            .iter()
            .position(|x| x.0 != "OFF" && x.0 != "IDLE" && x.0 != "DOWN")
            .unwrap_or(1);

        let total: f64 = residencies.iter().map(|x| x.1 as f64).sum();
        let active: f64 = residencies.iter().skip(offset).map(|x| x.1 as f64).sum();

        if total <= 0.0 {
            return (0.0, 0.0);
        }

        let usage_ratio = active / total;

        // Weighted average frequency.
        let freqs = &self.gpu_freqs;
        let active_states = &residencies[offset..];
        let mut avg_freq = 0.0f64;
        if !freqs.is_empty() && active > 0.0 {
            for (i, state) in active_states.iter().enumerate() {
                if i < freqs.len() {
                    let pct = state.1 as f64 / active;
                    avg_freq += pct * freqs[i] as f64;
                }
            }
        }

        (avg_freq, usage_ratio)
    }
}

impl Drop for IOReportSampler {
    fn drop(&mut self) {
        if let Some(prev) = self.prev_sample {
            unsafe { CFRelease(prev as CFTypeRef) };
        }
        // subscription and channels are owned by IOReport — don't release.
    }
}

fn energy_to_watts(val: f64, unit: &str) -> f64 {
    // IOReport energy values are cumulative over the delta interval.
    // For a ~1s sample interval, mJ ≈ mW numerically.
    match unit {
        "mJ" => val / 1e3,
        "uJ" => val / 1e6,
        "nJ" => val / 1e9,
        _ => val / 1e3, // default assume mJ
    }
}
