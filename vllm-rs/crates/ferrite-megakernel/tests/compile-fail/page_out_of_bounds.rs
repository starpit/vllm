// Page<5, 4, Empty>::new() — ID >= NUM_PAGES, must compile-fail.
use ferrite_megakernel::{Empty, Page};

fn main() {
    let _ = Page::<5, 4, Empty>::new();
}
