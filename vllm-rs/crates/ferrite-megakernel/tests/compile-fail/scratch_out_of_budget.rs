// ScratchRegion<8129, 64, 8192, RmsNormScope>::new() — OFFSET+BYTES > SCRATCH_BYTES.
use ferrite_megakernel::{RmsNormScope, ScratchRegion};

fn main() {
    let _ = ScratchRegion::<8129, 64, 8192, RmsNormScope>::new();
}
