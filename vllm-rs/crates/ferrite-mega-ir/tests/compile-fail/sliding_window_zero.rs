// SlidingWindow::<0>::new() — must compile-fail.
use ferrite_mega_ir::SlidingWindow;

fn main() {
    let _ = SlidingWindow::<0>::new();
}
