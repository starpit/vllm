// Two ScratchRegion<...>s in the same scope that overlap — disjoint_with must compile-fail.
use ferrite_mega_ir::{RmsNormScope, ScratchRegion};

fn main() {
    let a = ScratchRegion::<0, 128, 8192, RmsNormScope>::new();
    let b = ScratchRegion::<64, 64, 8192, RmsNormScope>::new();
    let _ = a.disjoint_with(b); // [0,128) and [64,128) overlap
}
