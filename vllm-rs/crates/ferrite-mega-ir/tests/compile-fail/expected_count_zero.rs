// ExpectedCount::<0>::new() — must compile-fail.
use ferrite_mega_ir::ExpectedCount;

fn main() {
    let _ = ExpectedCount::<0>::new();
}
