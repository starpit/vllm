// SubstrateBudget<0, ...>::new() — NUM_PAGES must be > 0, must compile-fail.
use ferrite_mega_ir::SubstrateBudget;

fn main() {
    let _ = SubstrateBudget::<0, 8, 32_768, 8_192, 0>::new();
}
