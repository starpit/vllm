// ExpectedCount::<0>::new() — must compile-fail.
use ferrite_megakernel::ExpectedCount;

fn main() {
    let _ = ExpectedCount::<0>::new();
}
