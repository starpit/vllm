// LayerIndex::<16, 16>::new() — must compile-fail.
use ferrite_megakernel::LayerIndex;

fn main() {
    let _ = LayerIndex::<16, 16>::new();
}
