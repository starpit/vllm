// MatmulShape<0, 16> — N must be > 0, must compile-fail.
use ferrite_mega_ir::MatmulShape;

fn main() {
    let _ = MatmulShape::<0, 16>::new();
}
