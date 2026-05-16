// LayerIndex::<16, 16>::new() — must compile-fail.
use ferrite_mega_ir::LayerIndex;

fn main() {
    let _ = LayerIndex::<16, 16>::new();
}
