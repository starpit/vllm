pub mod config;
pub mod smem;
pub mod tile;
pub mod gemm;
pub mod silu;
pub mod rmsnorm;
pub mod fused;

use std::fmt::Write;
use config::GemmConfig;

/// Register class in PTX.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegClass { Pred, B32, B64, F32 }

/// A typed register handle.
#[derive(Clone, Copy, Debug)]
pub struct Reg {
    pub class: RegClass,
    pub index: u32,
}

impl Reg {
    pub fn pred(i: u32) -> Self { Self { class: RegClass::Pred, index: i } }
    pub fn r(i: u32) -> Self { Self { class: RegClass::B32, index: i } }
    pub fn rd(i: u32) -> Self { Self { class: RegClass::B64, index: i } }
    pub fn f(i: u32) -> Self { Self { class: RegClass::F32, index: i } }
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.class {
            RegClass::Pred => write!(f, "%p{}", self.index),
            RegClass::B32 => write!(f, "%r{}", self.index),
            RegClass::B64 => write!(f, "%rd{}", self.index),
            RegClass::F32 => write!(f, "%f{}", self.index),
        }
    }
}

/// Tracks peak register usage per class.
pub struct RegAllocator {
    pred_next: u32,
    b32_next: u32,
    b64_next: u32,
    f32_next: u32,
}

impl RegAllocator {
    pub fn new() -> Self {
        Self { pred_next: 1, b32_next: 1, b64_next: 1, f32_next: 1 }
    }

    pub fn alloc_pred(&mut self) -> Reg {
        let r = Reg::pred(self.pred_next);
        self.pred_next += 1;
        r
    }

    pub fn alloc_b32(&mut self) -> Reg {
        let r = Reg::r(self.b32_next);
        self.b32_next += 1;
        r
    }

    pub fn alloc_b32_range(&mut self, count: u32) -> Vec<Reg> {
        (0..count).map(|_| self.alloc_b32()).collect()
    }

    pub fn alloc_b64(&mut self) -> Reg {
        let r = Reg::rd(self.b64_next);
        self.b64_next += 1;
        r
    }

    pub fn alloc_f32(&mut self) -> Reg {
        let r = Reg::f(self.f32_next);
        self.f32_next += 1;
        r
    }

    pub fn pred_count(&self) -> u32 { self.pred_next }
    pub fn b32_count(&self) -> u32 { self.b32_next }
    pub fn b64_count(&self) -> u32 { self.b64_next }
    pub fn f32_count(&self) -> u32 { self.f32_next }
}

/// PTX code builder — emits instructions as formatted strings.
pub struct PtxBuilder {
    pub config: GemmConfig,
    pub regs: RegAllocator,
    pub body: String,
}

impl PtxBuilder {
    pub fn new(config: GemmConfig) -> Self {
        Self {
            config,
            regs: RegAllocator::new(),
            body: String::with_capacity(32 * 1024),
        }
    }

    pub fn w(&mut self, s: &str) {
        writeln!(self.body, "\t{}", s).unwrap();
    }

    // ── Arithmetic ──

    pub fn add_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("add.s32 \t{d}, {a}, {b};"));
    }
    pub fn add_s32_imm(&mut self, d: Reg, a: Reg, imm: i32) {
        self.w(&format!("add.s32 \t{d}, {a}, {imm};"));
    }
    pub fn add_s64(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("add.s64 \t{d}, {a}, {b};"));
    }
    pub fn add_s64_imm(&mut self, d: Reg, a: Reg, imm: i64) {
        self.w(&format!("add.s64 \t{d}, {a}, {imm};"));
    }
    pub fn mov_b64(&mut self, d: Reg, s: Reg) {
        self.w(&format!("mov.b64 \t{d}, {s};"));
    }
    pub fn mad_wide_s32_imm(&mut self, d: Reg, a: Reg, imm: i32, c: Reg) {
        self.w(&format!("mad.wide.s32 \t{d}, {a}, {imm}, {c};"));
    }
    pub fn sub_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("sub.s32 \t{d}, {a}, {b};"));
    }
    pub fn mul_lo_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("mul.lo.s32 \t{d}, {a}, {b};"));
    }
    pub fn mul_wide_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("mul.wide.s32 \t{d}, {a}, {b};"));
    }
    pub fn mul_wide_u32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("mul.wide.u32 \t{d}, {a}, {b};"));
    }
    pub fn mad_wide_s32(&mut self, d: Reg, a: Reg, b: Reg, c: Reg) {
        self.w(&format!("mad.wide.s32 \t{d}, {a}, {b}, {c};"));
    }
    pub fn shl_b32(&mut self, d: Reg, a: Reg, bits: u32) {
        self.w(&format!("shl.b32 \t{d}, {a}, {bits};"));
    }
    pub fn shl_b64(&mut self, d: Reg, a: Reg, bits: u32) {
        self.w(&format!("shl.b64 \t{d}, {a}, {bits};"));
    }
    pub fn shr_u32(&mut self, d: Reg, a: Reg, bits: u32) {
        self.w(&format!("shr.u32 \t{d}, {a}, {bits};"));
    }
    pub fn and_b32(&mut self, d: Reg, a: Reg, mask: u32) {
        self.w(&format!("and.b32 \t{d}, {a}, {mask};"));
    }
    pub fn and_b32_reg(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("and.b32 \t{d}, {a}, {b};"));
    }
    pub fn or_b32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("or.b32 \t{d}, {a}, {b};"));
    }
    pub fn xor_b32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("xor.b32 \t{d}, {a}, {b};"));
    }
    pub fn xor_b32_imm(&mut self, d: Reg, a: Reg, imm: u32) {
        self.w(&format!("xor.b32 \t{d}, {a}, {imm};"));
    }
    pub fn bfe_u32(&mut self, d: Reg, a: Reg, start: u32, len: u32) {
        self.w(&format!("bfe.u32 \t{d}, {a}, {start}, {len};"));
    }
    pub fn mov_b32(&mut self, d: Reg, s: Reg) {
        self.w(&format!("mov.b32 \t{d}, {s};"));
    }
    pub fn mov_b32_imm(&mut self, d: Reg, val: u32) {
        self.w(&format!("mov.b32 \t{d}, {val};"));
    }
    pub fn mov_b32_name(&mut self, d: Reg, name: &str) {
        self.w(&format!("mov.u32 \t{d}, {name};"));
    }
    pub fn cvt_u64_u32(&mut self, d: Reg, s: Reg) {
        self.w(&format!("cvt.u64.u32 \t{d}, {s};"));
    }
    pub fn cvt_s64_s32(&mut self, d: Reg, s: Reg) {
        self.w(&format!("cvt.s64.s32 \t{d}, {s};"));
    }
    pub fn selp_b32(&mut self, d: Reg, a: u32, b: u32, p: Reg) {
        self.w(&format!("selp.b32 \t{d}, {a}, {b}, {p};"));
    }
    pub fn selp_b32_regs(&mut self, d: Reg, a: Reg, b: Reg, p: Reg) {
        self.w(&format!("selp.b32 \t{d}, {a}, {b}, {p};"));
    }
    pub fn selp_b32_imm_reg(&mut self, d: Reg, a: u32, b: Reg, p: Reg) {
        self.w(&format!("selp.b32 \t{d}, {a}, {b}, {p};"));
    }
    pub fn mov_f32_imm(&mut self, d: Reg, val: f32) {
        self.w(&format!("mov.b32 \t{d}, 0x{:08X};", val.to_bits()));
    }
    /// Move a .b32 register into a .f32 register (reinterpret bits).
    pub fn mov_b32_to_f32(&mut self, d: Reg, s: Reg) {
        self.w(&format!("mov.b32 \t{d}, {s};"));
    }
    /// Move a .f32 register into a .b32 register (reinterpret bits).
    pub fn mov_f32_to_b32(&mut self, d: Reg, s: Reg) {
        self.w(&format!("mov.b32 \t{d}, {s};"));
    }

    // ── Float arithmetic (f32) ──

    pub fn neg_f32(&mut self, d: Reg, a: Reg) {
        self.w(&format!("neg.f32 \t{d}, {a};"));
    }
    pub fn mul_f32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("mul.f32 \t{d}, {a}, {b};"));
    }
    pub fn mul_f32_imm(&mut self, d: Reg, a: Reg, imm: f32) {
        self.w(&format!("mul.f32 \t{d}, {a}, 0F{:08X};", imm.to_bits()));
    }
    pub fn add_f32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("add.f32 \t{d}, {a}, {b};"));
    }
    pub fn add_f32_imm(&mut self, d: Reg, a: Reg, imm: f32) {
        self.w(&format!("add.f32 \t{d}, {a}, 0F{:08X};", imm.to_bits()));
    }
    pub fn ex2_approx_f32(&mut self, d: Reg, a: Reg) {
        self.w(&format!("ex2.approx.f32 \t{d}, {a};"));
    }
    pub fn rcp_approx_f32(&mut self, d: Reg, a: Reg) {
        self.w(&format!("rcp.approx.f32 \t{d}, {a};"));
    }

    // ── Predicate ──

    pub fn setp_gt_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("setp.gt.s32 \t{d}, {a}, {b};"));
    }
    pub fn setp_gt_s32_imm(&mut self, d: Reg, a: Reg, imm: i32) {
        self.w(&format!("setp.gt.s32 \t{d}, {a}, {imm};"));
    }
    pub fn setp_lt_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("setp.lt.s32 \t{d}, {a}, {b};"));
    }

    // ── Memory ──

    pub fn ld_param_b64(&mut self, d: Reg, name: &str) {
        self.w(&format!("ld.param.b64 \t{d}, [{name}];"));
    }
    pub fn ld_param_b32(&mut self, d: Reg, name: &str) {
        self.w(&format!("ld.param.b32 \t{d}, [{name}];"));
    }
    pub fn ld_global_b32(&mut self, d: Reg, addr: Reg, offset: i32) {
        if offset == 0 {
            self.w(&format!("ld.global.b32 \t{d}, [{addr}];"));
        } else {
            self.w(&format!("ld.global.b32 \t{d}, [{addr}+{offset}];"));
        }
    }
    pub fn ld_global_f32(&mut self, d: Reg, addr: Reg, offset: i32) {
        if offset == 0 {
            self.w(&format!("ld.global.f32 \t{d}, [{addr}];"));
        } else {
            self.w(&format!("ld.global.f32 \t{d}, [{addr}+{offset}];"));
        }
    }
    pub fn st_global_f32(&mut self, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("st.global.f32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("st.global.f32 \t[{addr}+{offset}], {val};"));
        }
    }
    pub fn st_global_b32(&mut self, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("st.global.b32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("st.global.b32 \t[{addr}+{offset}], {val};"));
        }
    }

    // ── Vectorized loads ──

    /// ld.global.v4.b32 {d0, d1, d2, d3}, [addr+offset];
    /// Loads 16 bytes (4 x b32) from global memory.
    pub fn ld_global_v4_b32(&mut self, d: [Reg; 4], addr: Reg, offset: i32) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "ld.global.v4.b32 \t{{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    /// ld.global.v2.b32 {d0, d1}, [addr+offset];
    /// Loads 8 bytes (2 x b32) from global memory.
    pub fn ld_global_v2_b32(&mut self, d: [Reg; 2], addr: Reg, offset: i32) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "ld.global.v2.b32 \t{{{}, {}}}, {addr_str};",
            d[0], d[1]
        ));
    }

    /// Predicated ld.global.v4.b32.
    pub fn pred_ld_global_v4_b32(&mut self, pred: Reg, d: [Reg; 4], addr: Reg, offset: i32) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "@{pred} ld.global.v4.b32 \t{{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    // ── Shared memory stores ──

    /// st.shared.b32 [addr+offset], val;
    pub fn st_shared_b32(&mut self, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("st.shared.b32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("st.shared.b32 \t[{addr}+{offset}], {val};"));
        }
    }

    /// Predicated st.shared.b32.
    pub fn pred_st_shared_b32(&mut self, pred: Reg, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("@{pred} st.shared.b32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("@{pred} st.shared.b32 \t[{addr}+{offset}], {val};"));
        }
    }

    /// ld.shared.b32 d, [addr+offset];
    pub fn ld_shared_b32(&mut self, d: Reg, addr: Reg, offset: i32) {
        if offset == 0 {
            self.w(&format!("ld.shared.b32 \t{d}, [{addr}];"));
        } else {
            self.w(&format!("ld.shared.b32 \t{d}, [{addr}+{offset}];"));
        }
    }

    // ── Half-precision conversions ──

    /// cvt.f32.f16 d, src; -- convert the low 16 bits of src (b32) to f32.
    pub fn cvt_f32_f16(&mut self, d: Reg, src: Reg) {
        self.w(&format!("cvt.f32.f16 \t{d}, {src};"));
    }

    /// cvt.rn.f16.f32 d, src; -- convert f32 to f16 (round nearest).
    pub fn cvt_rn_f16_f32(&mut self, d: Reg, src: Reg) {
        self.w(&format!("cvt.rn.f16.f32 \t{d}, {src};"));
    }

    // ── FMA ──

    /// fma.rn.f32 d, a, b, c; -- d = a*b + c
    pub fn fma_f32(&mut self, d: Reg, a: Reg, b: Reg, c: Reg) {
        self.w(&format!("fma.rn.f32 \t{d}, {a}, {b}, {c};"));
    }

    // ── rsqrt ──

    pub fn rsqrt_approx_f32(&mut self, d: Reg, a: Reg) {
        self.w(&format!("rsqrt.approx.f32 \t{d}, {a};"));
    }

    // ── Shuffle ──

    /// shfl.sync.bfly.b32 d, src, offset, 0x1F, 0xFFFFFFFF;
    pub fn shfl_bfly(&mut self, d: Reg, src: Reg, offset: u32) {
        self.w(&format!(
            "shfl.sync.bfly.b32 \t{d}, {src}, {offset}, 0x1F, 0xFFFFFFFF;"
        ));
    }

    // ── Predicated operations ──

    /// @pred instruction (generic predication prefix)
    pub fn pred_bra_neg(&mut self, pred: Reg, label: &str) {
        self.w(&format!("@!{pred} bra \t{label};"));
    }

    pub fn setp_eq_s32(&mut self, d: Reg, a: Reg, imm: i32) {
        self.w(&format!("setp.eq.s32 \t{d}, {a}, {imm};"));
    }

    // ── Async copy ──

    pub fn cp_async_cg(&mut self, dst: Reg, dst_off: i32, src: Reg, src_off: i32, size_pred: Reg) {
        self.w(&format!(
            "cp.async.cg.shared.global [{dst}+{dst_off}], [{src}+{src_off}], 0x10, {size_pred};"
        ));
    }
    pub fn cp_async_commit(&mut self) {
        self.w("cp.async.commit_group;");
    }
    pub fn cp_async_wait_group(&mut self, n: u32) {
        self.w(&format!("cp.async.wait_group \t{n};"));
    }

    // ── Tensor core ──

    pub fn ldmatrix_x4(&mut self, d: [Reg; 4], addr: Reg, offset: Option<i32>) {
        let addr_str = match offset {
            Some(off) if off != 0 => format!("[{addr}+{off}]"),
            _ => format!("[{addr}]"),
        };
        self.w(&format!(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    pub fn ldmatrix_x4_trans(&mut self, d: [Reg; 4], addr: Reg, offset: Option<i32>) {
        let addr_str = match offset {
            Some(off) if off != 0 => format!("[{addr}+{off}]"),
            _ => format!("[{addr}]"),
        };
        self.w(&format!(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    pub fn mma_m16n8k16(&mut self, d: [Reg; 4], a: [Reg; 4], b: [Reg; 2], c: [Reg; 4]) {
        self.w(&format!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{}, {}, {}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}, {}, {}}};",
            d[0], d[1], d[2], d[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            c[0], c[1], c[2], c[3]
        ));
    }

    // ── Control flow ──

    pub fn bar_sync(&mut self, id: u32) {
        self.w(&format!("bar.sync \t{id};"));
    }
    pub fn label(&mut self, name: &str) {
        writeln!(self.body, "{name}:").unwrap();
    }
    pub fn bra_uni(&mut self, label: &str) {
        self.w(&format!("bra.uni \t{label};"));
    }
    pub fn pred_bra(&mut self, pred: Reg, label: &str) {
        self.w(&format!("@{pred} bra \t{label};"));
    }
    pub fn ret(&mut self) {
        self.w("ret;");
    }
    pub fn comment(&mut self, text: &str) {
        writeln!(self.body, "\t// {text}").unwrap();
    }
    pub fn blank(&mut self) {
        writeln!(self.body).unwrap();
    }

    // ── Finalize ──

    pub fn finalize(&self, kernel_name: &str, params: &[(& str, &str)]) -> String {
        let mut out = String::with_capacity(self.body.len() + 1024);
        writeln!(out, ".version 8.6").unwrap();
        writeln!(out, ".target {}", self.config.sm_arch).unwrap();
        writeln!(out, ".address_size 64").unwrap();
        writeln!(out).unwrap();
        writeln!(out, ".extern .shared .align 128 .b8 global_smem[];").unwrap();
        writeln!(out).unwrap();
        write!(out, ".visible .entry {kernel_name}(\n").unwrap();
        for (i, (ty, name)) in params.iter().enumerate() {
            let comma = if i + 1 < params.len() { "," } else { "" };
            writeln!(out, "\t.param {ty} {name}{comma}").unwrap();
        }
        writeln!(out, ")").unwrap();
        writeln!(out, ".reqntid {}", self.config.threads()).unwrap();
        writeln!(out, "{{").unwrap();
        writeln!(out, "\t.reg .pred \t%p<{}>;", self.regs.pred_count()).unwrap();
        writeln!(out, "\t.reg .b32 \t%r<{}>;", self.regs.b32_count()).unwrap();
        writeln!(out, "\t.reg .b64 \t%rd<{}>;", self.regs.b64_count()).unwrap();
        if self.regs.f32_count() > 1 {
            writeln!(out, "\t.reg .f32 \t%f<{}>;", self.regs.f32_count()).unwrap();
        }
        writeln!(out).unwrap();
        out.push_str(&self.body);
        writeln!(out).unwrap();
        writeln!(out, "}}").unwrap();
        out
    }
}
