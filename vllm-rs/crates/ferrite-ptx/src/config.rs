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
