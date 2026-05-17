// IterCount::<0>::new() — must compile-fail.
use ferrite_megakernel::IterCount;

fn main() {
    let _ = IterCount::<0>::new();
}
