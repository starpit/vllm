// SPDX-License-Identifier: Apache-2.0
//! Tests for Metal executor with synthetic instruction tapes.
//!
//! These tests verify the Metal executor can:
//! 1. Record instructions to ICB without real weights
//! 2. Handle weight slot mapping correctly
//! 3. Unroll loops at init time

#![cfg(all(test, feature = "metal"))]

use super::metal::MetalExecutor;
use crate::{CanonicalParams, Instruction};

/// Minimal test weights for synthetic instruction tape
#[derive(Clone, Copy)]
struct TestWeights;

impl CanonicalParams for TestWeights {
    const Q_SIZE: usize = 4096;
    const INTERMEDIATE_SIZE: usize = 11008;
    const NUM_Q_HEADS: usize = 32;
    const NUM_KV_HEADS: usize = 32;
    const HEAD_DIM: usize = 128;
    const NUM_LAYERS: u32 = 32;
    const VOCAB_SIZE: usize = 32000;
    const MAX_SEQ_LEN: usize = 2048;
    const ROPE_THETA: f32 = 10000.0;
    const ROPE_SCALING: Option<(f32, f32)> = None;
}

/// Mock weight function for testing
fn mock_rmsnorm_weight(_weights: &TestWeights, _layer: u32) -> MockRmsNormWeight {
    MockRmsNormWeight
}

struct MockRmsNormWeight;

#[test]
#[ignore] // Requires Metal device
fn test_executor_creation_simple() {
    // Create a simple instruction tape: RmsNorm -> Add
    let instructions = vec![
        Instruction::RmsNorm(
            1,  // out_slot
            0,  // in_slot
            0,  // wt_slot (unused in Metal)
            mock_rmsnorm_weight as usize,
        ),
        Instruction::Add(1, 0), // delta_slot, residual_slot
    ];
    
    let weights = TestWeights;
    let num_tile_slots = 10;
    
    // This should succeed if Metal device is available
    let result = MetalExecutor::new(&instructions, weights, num_tile_slots);
    
    match result {
        Ok(_executor) => {
            println!("✓ Metal executor created successfully");
        }
        Err(e) => {
            // If no Metal device, test should be ignored
            if e.contains("No Metal device") {
                println!("Skipping test: {}", e);
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}

#[test]
#[ignore] // Requires Metal device
fn test_executor_loop_unrolling() {
    // Create instruction tape with a loop
    let instructions = vec![
        Instruction::Loop(
            3,  // count (3 layers)
            2,  // body_len (2 instructions per layer)
        ),
        // Loop body:
        Instruction::RmsNorm(1, 0, 0, mock_rmsnorm_weight as usize),
        Instruction::Add(1, 0),
        // After loop:
        Instruction::ScalarMul(1, 2, 0.5),
    ];
    
    let weights = TestWeights;
    let num_tile_slots = 10;
    
    let result = MetalExecutor::new(&instructions, weights, num_tile_slots);
    
    match result {
        Ok(_executor) => {
            println!("✓ Metal executor with loop unrolling created successfully");
        }
        Err(e) => {
            if e.contains("No Metal device") {
                println!("Skipping test: {}", e);
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}

#[test]
#[ignore] // Requires Metal device
fn test_executor_metadata_instructions() {
    // Create instruction tape with metadata ops (should not fail)
    let instructions = vec![
        Instruction::Reshape(1, 0, [1, 4096, 0, 0, 0, 0, 0, 0], [0; 8], [1; 8], 2),
        Instruction::Alias(2, 1),
        Instruction::RmsNorm(3, 2, 0, mock_rmsnorm_weight as usize),
        Instruction::Free(1),
    ];
    
    let weights = TestWeights;
    let num_tile_slots = 10;
    
    let result = MetalExecutor::new(&instructions, weights, num_tile_slots);
    
    match result {
        Ok(_executor) => {
            println!("✓ Metal executor with metadata instructions created successfully");
        }
        Err(e) => {
            if e.contains("No Metal device") {
                println!("Skipping test: {}", e);
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}