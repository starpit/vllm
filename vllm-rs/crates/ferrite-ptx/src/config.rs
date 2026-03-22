/// All parameters that define a GEMM kernel shape.
#[derive(Clone, Debug)]
pub struct GemmConfig {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub wm: u32,
    pub wn: u32,
    pub mma_m: u32,
    pub mma_n: u32,
    pub mma_k: u32,
    pub num_stages: u32,
    pub sm_arch: String,
}

impl GemmConfig {
    pub fn default_64x64() -> Self {
        Self {
            bm: 64,
            bn: 64,
            bk: 32,
            wm: 64,
            wn: 16,
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 2,
            sm_arch: "sm_89".into(),
        }
    }

    /// 128×64 tile — used for dual GEMM (matches CUTLASS examples/45_dual_gemm).
    /// Halved N-dimension reduces accumulator pressure for dual accumulators.
    /// 4 warps (2×2 layout), 128 threads.
    pub fn default_128x64() -> Self {
        Self {
            bm: 128,
            bn: 64,
            bk: 32,
            wm: 64,
            wn: 32, // 2×2 warp layout: warps_m=2, warps_n=2
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 2, // 3-stage buffer cycling bug still present
            sm_arch: "sm_89".into(),
        }
    }

    pub fn default_128x128() -> Self {
        Self {
            bm: 128,
            bn: 128,
            bk: 32,
            wm: 64,
            wn: 64, // 2×2 warp layout
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 2,
            sm_arch: "sm_89".into(),
        }
    }

    pub fn reg_m(&self) -> u32 {
        self.wm / self.mma_m
    }
    pub fn reg_n(&self) -> u32 {
        self.wn / self.mma_n
    }
    pub fn k_iters(&self) -> u32 {
        self.bk / self.mma_k
    }
    pub fn warps_m(&self) -> u32 {
        self.bm / self.wm
    }
    pub fn warps_n(&self) -> u32 {
        self.bn / self.wn
    }
    pub fn warps(&self) -> u32 {
        self.warps_m() * self.warps_n()
    }
    pub fn threads(&self) -> u32 {
        self.warps() * 32
    }
    pub fn num_acc(&self) -> u32 {
        self.reg_m() * self.reg_n() * 4
    }
    pub fn smem_a_bytes(&self) -> u32 {
        self.bm * self.bk * 2
    }
    pub fn smem_b_bytes(&self) -> u32 {
        self.bk * self.bn * 2
    }
    pub fn buf_stride(&self) -> u32 {
        self.smem_a_bytes() + self.smem_b_bytes()
    }
    pub fn smem_total(&self) -> u32 {
        self.buf_stride() * self.num_stages
    }

    /// Number of cp.async 16-byte chunks needed to load one A tile
    pub fn cp_chunks_a(&self) -> u32 {
        self.smem_a_bytes() / (self.threads() * 16)
    }
    /// Number of cp.async 16-byte chunks needed to load one B tile
    pub fn cp_chunks_b(&self) -> u32 {
        self.smem_b_bytes() / (self.threads() * 16)
    }

    /// Number of B column groups for ldmatrix.trans.
    /// Each column group covers 4 rn values (XOR 0,32,64,96).
    /// For BN=64 (WN=16, REG_N=2): 1 group of 2 rn (uses 2 of 4 XOR offsets).
    /// For BN=128 (WN=64, REG_N=8): 2 groups of 4 rn each.
    pub fn b_col_groups(&self) -> u32 {
        let rn = self.reg_n();
        (rn + 3) / 4 // ceil(reg_n / 4)
    }

    /// Byte stride between B column groups in ldmatrix.trans addressing.
    /// Each group covers 4 × MMA_N = 32 columns = 32 * BK * 2 bytes in smem.
    /// But with swizzled layout, the stride is 128 bytes per group.
    pub fn b_col_group_stride(&self) -> u32 {
        128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════
    // default_64x64 derived values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_64x64_reg_m() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.reg_m(), 4, "64x64: reg_m = wm/mma_m = 64/16 = 4");
    }

    #[test]
    fn test_64x64_reg_n() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.reg_n(), 2, "64x64: reg_n = wn/mma_n = 16/8 = 2");
    }

    #[test]
    fn test_64x64_k_iters() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.k_iters(), 2, "64x64: k_iters = bk/mma_k = 32/16 = 2");
    }

    #[test]
    fn test_64x64_warps() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.warps(), 4, "64x64: warps = (64/64) * (64/16) = 1 * 4 = 4");
    }

    #[test]
    fn test_64x64_threads() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.threads(), 128, "64x64: threads = 4 warps * 32 = 128");
    }

    #[test]
    fn test_64x64_num_acc() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.num_acc(), 32, "64x64: num_acc = 4 * 2 * 4 = 32");
    }

    #[test]
    fn test_64x64_smem_a_bytes() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.smem_a_bytes(), 4096, "64x64: smem_a = 64 * 32 * 2 = 4096");
    }

    #[test]
    fn test_64x64_smem_b_bytes() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.smem_b_bytes(), 4096, "64x64: smem_b = 32 * 64 * 2 = 4096");
    }

    #[test]
    fn test_64x64_smem_total() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.smem_total(), 16384, "64x64: smem_total = (4096 + 4096) * 2 = 16384");
    }

    #[test]
    fn test_64x64_cp_chunks() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.cp_chunks_a(), 2, "64x64: cp_chunks_a = 4096 / (128 * 16) = 2");
        assert_eq!(c.cp_chunks_b(), 2, "64x64: cp_chunks_b = 4096 / (128 * 16) = 2");
    }

    // ═══════════════════════════════════════════════════════════════════
    // default_128x128 derived values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_128x128_reg_m() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.reg_m(), 4, "128x128: reg_m = wm/mma_m = 64/16 = 4");
    }

    #[test]
    fn test_128x128_reg_n() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.reg_n(), 8, "128x128: reg_n = wn/mma_n = 64/8 = 8");
    }

    #[test]
    fn test_128x128_k_iters() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.k_iters(), 2, "128x128: k_iters = bk/mma_k = 32/16 = 2");
    }

    #[test]
    fn test_128x128_warps() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.warps(), 4, "128x128: warps = (128/64) * (128/64) = 2 * 2 = 4");
    }

    #[test]
    fn test_128x128_threads() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.threads(), 128, "128x128: threads = 4 warps * 32 = 128");
    }

    #[test]
    fn test_128x128_num_acc() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.num_acc(), 128, "128x128: num_acc = 4 * 8 * 4 = 128");
    }

    #[test]
    fn test_128x128_smem_a_bytes() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.smem_a_bytes(), 8192, "128x128: smem_a = 128 * 32 * 2 = 8192");
    }

    #[test]
    fn test_128x128_smem_b_bytes() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.smem_b_bytes(), 8192, "128x128: smem_b = 32 * 128 * 2 = 8192");
    }

    #[test]
    fn test_128x128_smem_total() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.smem_total(), 32768, "128x128: smem_total = (8192 + 8192) * 2 = 32768");
    }

    #[test]
    fn test_128x128_cp_chunks() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.cp_chunks_a(), 4, "128x128: cp_chunks_a = 8192 / (128 * 16) = 4");
        assert_eq!(c.cp_chunks_b(), 4, "128x128: cp_chunks_b = 8192 / (128 * 16) = 4");
    }

    #[test]
    fn test_128x128_b_col_groups() {
        let c = GemmConfig::default_128x128();
        assert_eq!(c.b_col_groups(), 2, "128x128: b_col_groups = ceil(8/4) = 2");
    }

    #[test]
    fn test_64x64_b_col_groups() {
        let c = GemmConfig::default_64x64();
        assert_eq!(c.b_col_groups(), 1, "64x64: b_col_groups = ceil(2/4) = 1");
    }

    #[test]
    fn test_buf_stride() {
        let c64 = GemmConfig::default_64x64();
        assert_eq!(c64.buf_stride(), 8192, "64x64: buf_stride = 4096 + 4096 = 8192");

        let c128 = GemmConfig::default_128x128();
        assert_eq!(c128.buf_stride(), 16384, "128x128: buf_stride = 8192 + 8192 = 16384");
    }

    #[test]
    fn test_warps_m_and_n() {
        let c64 = GemmConfig::default_64x64();
        assert_eq!(c64.warps_m(), 1, "64x64: warps_m = 64/64 = 1");
        assert_eq!(c64.warps_n(), 4, "64x64: warps_n = 64/16 = 4");

        let c128 = GemmConfig::default_128x128();
        assert_eq!(c128.warps_m(), 2, "128x128: warps_m = 128/64 = 2");
        assert_eq!(c128.warps_n(), 2, "128x128: warps_n = 128/64 = 2");
    }
}
