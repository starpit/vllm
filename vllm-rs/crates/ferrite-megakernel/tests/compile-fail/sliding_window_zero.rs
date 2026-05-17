// SlidingWindow::<0>::new() — must compile-fail.
use ferrite_megakernel::SlidingWindow;

fn main() {
    let _ = SlidingWindow::<0>::new();
}
