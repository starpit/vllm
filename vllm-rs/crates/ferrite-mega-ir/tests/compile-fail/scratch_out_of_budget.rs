// ScratchRegion<8129, 64, 8192, RmsNormScope>::new() — OFFSET+BYTES > SCRATCH_BYTES.
use ferrite_mega_ir::{RmsNormScope, ScratchRegion};

fn main() {
    let _ = ScratchRegion::<8129, 64, 8192, RmsNormScope>::new();
}
