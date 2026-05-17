// RmsNorm::new with CONSUMER_BAR_REDUCE = 0 — bar 0 is reserved for
// __syncthreads. The sealed `BarSyncId<ID>: IsValidBarSyncId` witness
// has no impl for ID=0, so the `where` bound on `RmsNorm::new` fails
// at type-check (E0277), NOT at monomorphization assert.
use ferrite_megakernel::{FiniteF32, RmsNorm, WeightRef};

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
        /*CONSUMER_BAR_REDUCE=*/ 0,  // INVALID: bar 0 reserved
        /*CONSUMER_BAR_PUBLISH=*/ 2,
    >(
        WeightRef::new("W::norm".to_string()),
        FiniteF32::new(1.0e-5_f32),
    );
}
