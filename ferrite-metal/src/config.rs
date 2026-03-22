/// Metal GEMM kernel configuration.
///
/// Maps to MFA's GEMMKernelDescriptor. Tile sizes, precision, async strategy,
/// transpose state — all the parameters that control MSL codegen.
///
/// Reference: ~/git/ccv/lib/nnc/mfa/kernels/GEMMKernelDescriptor.hpp

/// Operand precision for Metal shader types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    FP16,
    BF16,
    FP32,
}

impl Precision {
    /// MSL type name.
    pub fn msl_name(&self) -> &'static str {
        match self {
            Precision::FP16 => "half",
            Precision::BF16 => "bfloat",
            Precision::FP32 => "float",
        }
    }

    /// Bytes per element.
    pub fn bytes(&self) -> u16 {
        match self {
            Precision::FP16 | Precision::BF16 => 2,
            Precision::FP32 => 4,
        }
    }
}

/// Precision configuration for A, B, C operands and optional bias.
#[derive(Clone, Debug)]
pub struct OperandPrecisions {
    pub a: Precision,
    pub b: Precision,
    pub c: Precision,
    pub bias: Precision,
}

impl Default for OperandPrecisions {
    fn default() -> Self {
        Self {
            a: Precision::FP16,
            b: Precision::FP16,
            c: Precision::FP32,
            bias: Precision::FP32,
        }
    }
}

/// GPU generation for async copy strategy selection.
///
/// apple9 (M3+) prefers simdgroup_event async_copy.
/// Older GPUs prefer direct threadgroup loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuGeneration {
    /// M1, M2 — prefer direct loads (async copy has overhead)
    Apple8,
    /// M3+ — prefer async copy (hardware-accelerated)
    Apple9,
}

/// Full GEMM kernel descriptor.
///
/// Maps to MFA's GEMMKernelDescriptor.
#[derive(Clone, Debug)]
pub struct MetalGemmConfig {
    /// Block dimensions: (M, N, K).
    /// apple9 f16: typically (32, 32, 8)
    /// older f16: typically (48, 48, 32)
    pub block_m: u16,
    pub block_n: u16,
    pub block_k: u16,

    /// Leading block dimensions for threadgroup memory padding.
    /// Controls bank conflict avoidance.
    /// None = auto-compute from block dimensions.
    pub leading_block_dims: Option<[u16; 3]>,

    /// Memory (device/threadgroup) precisions.
    pub memory_precisions: OperandPrecisions,

    /// Register (simdgroup_matrix) precisions.
    pub register_precisions: OperandPrecisions,

    /// Transpose state for A, B, C.
    /// false = row-major, true = column-major.
    pub transpose: [bool; 3],

    /// Use simdgroup_event async_copy for device→threadgroup loads.
    /// true on apple9 (M3+), false on older.
    pub prefer_async_load: bool,

    /// Use simdgroup_event async_copy for threadgroup→device stores.
    pub prefer_async_store: bool,

    /// Whether to disable async copy entirely (fallback).
    pub disable_async_copy: bool,

    /// Simdgroup splits: how many simdgroups along M and N.
    pub splits: [u16; 2],

    /// Whether to use bias.
    pub use_bias: bool,

    /// GPU generation (affects tile selection and async strategy).
    pub gpu_gen: GpuGeneration,
}

impl MetalGemmConfig {
    /// Default config for f16 GEMM on apple9 (M3+).
    pub fn default_apple9_f16() -> Self {
        Self {
            block_m: 32,
            block_n: 32,
            block_k: 8,
            leading_block_dims: Some([32, 32, 32]),
            memory_precisions: OperandPrecisions::default(),
            register_precisions: OperandPrecisions {
                a: Precision::FP16,
                b: Precision::FP16,
                c: Precision::FP32,
                bias: Precision::FP32,
            },
            transpose: [false, true, false], // A row-major, B col-major
            prefer_async_load: true,
            prefer_async_store: false,
            disable_async_copy: false,
            splits: [1, 1],
            use_bias: false,
            gpu_gen: GpuGeneration::Apple9,
        }
    }

    /// Default config for f16 GEMM on older (M1/M2).
    pub fn default_apple8_f16() -> Self {
        Self {
            block_m: 48,
            block_n: 48,
            block_k: 32,
            leading_block_dims: None,
            memory_precisions: OperandPrecisions::default(),
            register_precisions: OperandPrecisions {
                a: Precision::FP16,
                b: Precision::FP16,
                c: Precision::FP32,
                bias: Precision::FP32,
            },
            transpose: [false, true, false],
            prefer_async_load: false,
            prefer_async_store: false,
            disable_async_copy: false,
            splits: [1, 1],
            use_bias: false,
            gpu_gen: GpuGeneration::Apple8,
        }
    }

    /// Register tile dimensions per simdgroup.
    /// MFA: registerM = blockM / splits.y, registerN = blockN / splits.x
    pub fn register_m(&self) -> u16 {
        self.block_m / self.splits[1]
    }

    pub fn register_n(&self) -> u16 {
        self.block_n / self.splits[0]
    }

    /// Leading block dimension for an operand (with padding for bank conflicts).
    pub fn leading_block_dim(&self, operand: char) -> u16 {
        if let Some(dims) = self.leading_block_dims {
            match operand {
                'A' => dims[0],
                'B' => dims[1],
                'C' => dims[2],
                _ => 0,
            }
        } else {
            // Auto-compute: leading dimension = the non-K dimension
            match operand {
                'A' => if self.transpose[0] { self.block_m } else { self.block_k },
                'B' => if self.transpose[1] { self.block_k } else { self.block_n },
                'C' => self.block_n,
                _ => 0,
            }
        }
    }

    /// Threadgroup memory bytes for one tile of an operand.
    pub fn block_bytes(&self, operand: char) -> u32 {
        let lead = self.leading_block_dim(operand) as u32;
        let trail = match operand {
            'A' => if self.transpose[0] { self.block_k } else { self.block_m },
            'B' => if self.transpose[1] { self.block_n } else { self.block_k },
            'C' => self.block_m,
            _ => 0,
        } as u32;
        let prec = match operand {
            'A' => self.memory_precisions.a.bytes(),
            'B' => self.memory_precisions.b.bytes(),
            'C' => self.memory_precisions.c.bytes(),
            _ => 0,
        } as u32;
        lead * trail * prec
    }

    /// Total threadgroup memory allocation.
    /// max(A_bytes + B_bytes, C_bytes) — A+B shared with C.
    pub fn threadgroup_memory(&self) -> u32 {
        let ab = self.block_bytes('A') + self.block_bytes('B');
        let c = self.block_bytes('C');
        ab.max(c)
    }

    /// Number of threads per threadgroup.
    pub fn threadgroup_size(&self) -> u32 {
        // One simdgroup = 32 threads. Total simdgroups = splits[0] * splits[1].
        (self.splits[0] as u32) * (self.splits[1] as u32) * 32
    }
}
