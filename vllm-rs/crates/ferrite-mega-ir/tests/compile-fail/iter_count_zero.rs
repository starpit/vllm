// IterCount::<0>::new() — must compile-fail.
use ferrite_mega_ir::IterCount;

fn main() {
    let _ = IterCount::<0>::new();
}
