// SPDX-License-Identifier: Apache-2.0
//! Rust-side mirror of `csrc/instruction.cuh::instruction_t`. Field
//! order MUST match byte-for-byte; the generated `config.cuh` includes
//! `static_assert(sizeof(instruction_t) == 256)` and the const-eval
//! `assert!(size_of::<Instruction>() == 256)` in this file is the
//! Rust-side bookend.

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorRef {
    pub tensor_kind: u32,
    pub layer_or_index: u32,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BarrierRef {
    pub index: i32,
    pub expected_arrives: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Instruction {
    pub opcode: u16,
    pub layer_idx: u16,
    pub _pad0: u32,
    pub src: [TensorRef; 4],
    pub dst: [TensorRef; 2],
    pub indices: [i32; 16],
    pub src_barriers: [BarrierRef; 2],
    pub dst_barriers: [BarrierRef; 2],
    pub _pad1: [u8; 56],
}

const _: () = {
    assert!(
        core::mem::size_of::<Instruction>() == 256,
        "Instruction MUST be exactly 256 bytes — see csrc/instruction.cuh::instruction_t.",
    );
};

pub const OPCODE_NOOP: u16 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_is_256_bytes() {
        assert_eq!(core::mem::size_of::<Instruction>(), 256);
    }

    #[test]
    fn instruction_is_16_aligned() {
        assert_eq!(core::mem::align_of::<Instruction>(), 16);
    }
}
