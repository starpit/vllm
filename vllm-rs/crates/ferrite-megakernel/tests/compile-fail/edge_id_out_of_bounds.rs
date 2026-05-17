// EdgeId::<4, 4>::new() — must compile-fail.
use ferrite_megakernel::EdgeId;

fn main() {
    let _ = EdgeId::<4, 4>::new();
}
