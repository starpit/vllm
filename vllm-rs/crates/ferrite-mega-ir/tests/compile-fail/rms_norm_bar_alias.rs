// RmsNorm::new with CONSUMER_BAR_REDUCE == CONSUMER_BAR_PUBLISH —
// within-op bar alias would deadlock at runtime (consumer waits on
// the same bar it just arrived on, with mismatched expected counts).
// The sealed `BarSyncPair<A, B>: IsDistinctBarPair` witness has impls
// only for ordered pairs where A != B. Same value for both fails the
// where bound at type-check (E0277).
use ferrite_mega_ir::{FiniteF32, RmsNorm, WeightRef};

fn main() {
    let _ = RmsNorm::new::<
        /*IN_ID=*/ 0,
        /*WEIGHT_ID=*/ 1,
        /*PARTIAL_OFF=*/ 0,
        /*PARTIAL_BYTES=*/ 32,
        /*CONSUMER_PHASE=*/ 0,
        /*STORER_PHASE=*/ 1,
        /*LAYER=*/ 0,
        /*NUM_PAGES=*/ 8,
        /*NUM_LAYERS=*/ 16,
        /*SCRATCH_BYTES=*/ 8192,
        /*ARRIVES=*/ 0,
        /*HIDDEN_DIM=*/ 2048,
        /*NUM_TOKENS=*/ 8,
        /*IN_ACT_SLOT=*/ 0,
        /*OUT_ACT_SLOT=*/ 1,
        /*WEIGHT_ACCESSOR_IDX=*/ 0,
        /*CONSUMER_BAR_REDUCE=*/ 3,
        /*CONSUMER_BAR_PUBLISH=*/ 3,  // INVALID: alias with REDUCE
    >(
        WeightRef::new("W::norm".to_string()),
        FiniteF32::new(1.0e-5_f32),
    );
}
