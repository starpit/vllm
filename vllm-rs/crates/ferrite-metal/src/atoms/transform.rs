use super::TransformAtom;

/// Identity transform — no modification to A_mat.
pub struct IdentityTransform;

impl TransformAtom for IdentityTransform {
    fn is_identity(&self) -> bool {
        true
    }
}
