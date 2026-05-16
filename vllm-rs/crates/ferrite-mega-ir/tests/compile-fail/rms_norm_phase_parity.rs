// RmsNorm::new with CONSUMER_PHASE != ARRIVES & 1 — must compile-fail.
use ferrite_mega_ir::{RmsNorm, WeightRef};

fn main() {
    let _ = RmsNorm::new::<
        /*IN_ID=*/ 0,
        /*WEIGHT_ID=*/ 1,
        /*PARTIAL_OFF=*/ 0,
        /*PARTIAL_BYTES=*/ 32,
        /*CONSUMER_PHASE=*/ 1,  // wrong: ARRIVES=0, expects 0
        /*STORER_PHASE=*/ 0,    // wrong: ARRIVES+1=1, expects 1
        /*LAYER=*/ 0,
        /*NUM_PAGES=*/ 8,
        /*NUM_LAYERS=*/ 16,
        /*SCRATCH_BYTES=*/ 8192,
        /*ARRIVES=*/ 0,
    >(WeightRef::new("W::norm".to_string()));
}
