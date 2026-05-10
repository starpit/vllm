// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal device detection and management.

use ferrite_metal_targets::MetalTargetProfile;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Wrapper around Metal device with target profile
#[derive(Clone)]
pub struct MetalDevice {
    pub device: Device,
    pub profile: MetalTargetProfile,
    pub queue: CommandQueue,
}

impl MetalDevice {
    pub fn new(device: Device, profile: MetalTargetProfile) -> Self {
        let queue = device.newCommandQueue().expect("newCommandQueue returned nil");
        Self {
            device,
            profile,
            queue,
        }
    }
}

/// Detect the current Metal device and return appropriate profile
pub fn detect_device() -> Option<MetalDevice> {
    let device = MTLCreateSystemDefaultDevice()?;

    // Detect architecture from device name
    let name = device.name().to_string();
    let profile = if name.contains("M1") {
        ferrite_metal_targets::M1_8CORE
    } else if name.contains("M2") {
        ferrite_metal_targets::M2_10CORE
    } else if name.contains("M3") {
        ferrite_metal_targets::M3_10CORE
    } else if name.contains("M4") {
        ferrite_metal_targets::M4_10CORE
    } else {
        // Default to M1 for unknown devices
        eprintln!(
            "Warning: Unknown Metal device '{}', defaulting to M1 profile",
            name
        );
        ferrite_metal_targets::M1_8CORE
    };

    Some(MetalDevice::new(device, profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        if let Some(device) = detect_device() {
            println!("Detected device: {}", device.device.name());
            println!("Profile: {:?}", device.profile.generation);
            assert!(device.profile.gpu_cores > 0);
        }
    }
}
