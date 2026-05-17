// PageId<16, 16>::new() — ID == NUM_PAGES, must compile-fail.
use ferrite_megakernel::PageId;

fn main() {
    let _ = PageId::<16, 16>::new();
}
