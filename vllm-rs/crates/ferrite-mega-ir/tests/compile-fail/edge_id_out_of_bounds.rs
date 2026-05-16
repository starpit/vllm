// EdgeId::<4, 4>::new() — must compile-fail.
use ferrite_mega_ir::EdgeId;

fn main() {
    let _ = EdgeId::<4, 4>::new();
}
