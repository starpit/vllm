// MbarrierPhase::<0>::assert_matches::<5>() — 0 != 5&1==1, must compile-fail.
use ferrite_mega_ir::MbarrierPhase;

fn main() {
    let _ = MbarrierPhase::<0>::assert_matches::<5>();
}
