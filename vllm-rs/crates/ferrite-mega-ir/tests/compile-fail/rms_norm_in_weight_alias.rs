// RmsNorm::new with IN_ID == WEIGHT_ID — within-op alias must compile-fail.
use ferrite_mega_ir::{RmsNorm, WeightRef};

fn main() {
    let _ = RmsNorm::new::<
        /*IN_ID=*/ 1,
        /*WEIGHT_ID=*/ 1,  // ALIAS
        /*PARTIAL_OFF=*/ 0,
        /*PARTIAL_BYTES=*/ 32,
        /*CONSUMER_PHASE=*/ 0,
        /*STORER_PHASE=*/ 1,
        /*LAYER=*/ 0,
        /*NUM_PAGES=*/ 8,
        /*NUM_LAYERS=*/ 16,
        /*SCRATCH_BYTES=*/ 8192,
        /*ARRIVES=*/ 0,
    >(WeightRef::new("W::norm".to_string()));
}
