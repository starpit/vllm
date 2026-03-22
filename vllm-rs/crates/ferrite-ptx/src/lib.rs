pub mod atoms;
pub mod config;
pub mod convert;
pub mod dual_pipeline;
pub mod fused;
pub mod gelu;
pub mod gemm;
pub mod pipeline;
pub mod rmsnorm;
pub mod silu;
pub mod silu_mul_epilogue;
pub mod smem;
pub mod tile;

use config::GemmConfig;
use std::fmt::Write;

/// Register class in PTX.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegClass {
    Pred,
    B32,
    B64,
    F32,
}

/// Register prefix letters per scope depth.
/// Outer (0): %r, %rd, %f, %p
/// Scope 1:   %t, %td, %tf, %tp
/// Scope 2:   %u, %ud, %uf, %up
/// Scope 3:   %v, %vd, %vf, %vp
/// Scope 4+:  %w, %wd, %wf, %wp (etc.)
const SCOPE_PREFIXES: &[&[&str]] = &[
    // [pred, b32, b64, f32]
    &["p", "r", "rd", "f"],   // scope 0 (outer)
    &["tp", "t", "td", "tf"], // scope 1
    &["up", "u", "ud", "uf"], // scope 2
    &["vp", "v", "vd", "vf"], // scope 3
    &["wp", "w", "wd", "wf"], // scope 4
    &["xp", "x", "xd", "xf"], // scope 5
];

fn scope_prefix(scope_id: u16, class: RegClass) -> &'static str {
    let idx = scope_id as usize;
    assert!(
        idx < SCOPE_PREFIXES.len(),
        "Too many nested scopes (max {})",
        SCOPE_PREFIXES.len()
    );
    match class {
        RegClass::Pred => SCOPE_PREFIXES[idx][0],
        RegClass::B32 => SCOPE_PREFIXES[idx][1],
        RegClass::B64 => SCOPE_PREFIXES[idx][2],
        RegClass::F32 => SCOPE_PREFIXES[idx][3],
    }
}

/// A typed register handle with scope information.
#[derive(Clone, Copy, Debug)]
pub struct Reg {
    pub class: RegClass,
    pub index: u32,
    /// Scope depth: 0 = outer (kernel-level), 1+ = block-scoped.
    pub scope_id: u16,
}

impl Reg {
    pub fn pred(i: u32) -> Self {
        Self {
            class: RegClass::Pred,
            index: i,
            scope_id: 0,
        }
    }
    pub fn r(i: u32) -> Self {
        Self {
            class: RegClass::B32,
            index: i,
            scope_id: 0,
        }
    }
    pub fn rd(i: u32) -> Self {
        Self {
            class: RegClass::B64,
            index: i,
            scope_id: 0,
        }
    }
    pub fn f(i: u32) -> Self {
        Self {
            class: RegClass::F32,
            index: i,
            scope_id: 0,
        }
    }

    fn with_scope(class: RegClass, index: u32, scope_id: u16) -> Self {
        Self {
            class,
            index,
            scope_id,
        }
    }
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = scope_prefix(self.scope_id, self.class);
        write!(f, "%{}{}", prefix, self.index)
    }
}

/// Saved register allocator state for scope push/pop (legacy soft scoping).
#[derive(Clone, Debug)]
struct ScopeState {
    pred_next: u32,
    b32_next: u32,
    b64_next: u32,
    f32_next: u32,
}

/// Tracks peak register usage per class.
///
/// Supports `push_scope()` / `pop_scope()` for soft scoping (register name
/// reuse without PTX block boundaries) and `begin_scope()` / `end_scope()`
/// for native PTX block scoping with `{ .reg ... }`.
///
/// Native block scoping gives ptxas explicit register lifetime information:
/// registers declared inside `{ }` are guaranteed dead at the closing brace.
/// This allows ptxas to reuse physical registers across phases (e.g., norm
/// reduction phase and K-loop phase share the same hardware registers).
pub struct RegAllocator {
    pred_next: u32,
    b32_next: u32,
    b64_next: u32,
    f32_next: u32,
    // High-water marks for .reg declarations (outer scope only)
    pred_hwm: u32,
    b32_hwm: u32,
    b64_hwm: u32,
    f32_hwm: u32,
    // Soft scope stack (legacy push_scope/pop_scope)
    scope_stack: Vec<ScopeState>,
    // Current native block scope depth (0 = outer)
    block_scope_depth: u16,
}

impl RegAllocator {
    pub fn new() -> Self {
        Self {
            pred_next: 1,
            b32_next: 1,
            b64_next: 1,
            f32_next: 1,
            pred_hwm: 1,
            b32_hwm: 1,
            b64_hwm: 1,
            f32_hwm: 1,
            scope_stack: Vec::new(),
            block_scope_depth: 0,
        }
    }

    /// Save the current allocation counters (legacy soft scoping).
    /// Registers allocated after this call and before the matching `pop_scope()`
    /// will have their names freed for reuse.
    pub fn push_scope(&mut self) {
        self.pred_hwm = self.pred_hwm.max(self.pred_next);
        self.b32_hwm = self.b32_hwm.max(self.b32_next);
        self.b64_hwm = self.b64_hwm.max(self.b64_next);
        self.f32_hwm = self.f32_hwm.max(self.f32_next);
        self.scope_stack.push(ScopeState {
            pred_next: self.pred_next,
            b32_next: self.b32_next,
            b64_next: self.b64_next,
            f32_next: self.f32_next,
        });
    }

    /// Restore allocation counters (legacy soft scoping).
    pub fn pop_scope(&mut self) {
        self.pred_hwm = self.pred_hwm.max(self.pred_next);
        self.b32_hwm = self.b32_hwm.max(self.b32_next);
        self.b64_hwm = self.b64_hwm.max(self.b64_next);
        self.f32_hwm = self.f32_hwm.max(self.f32_next);
        let state = self
            .scope_stack
            .pop()
            .expect("pop_scope() called without matching push_scope()");
        self.pred_next = state.pred_next;
        self.b32_next = state.b32_next;
        self.b64_next = state.b64_next;
        self.f32_next = state.f32_next;
    }

    /// Enter a new native block scope. Resets counters to 1 and increments scope depth.
    /// Returns the new scope depth.
    fn enter_block_scope(&mut self) -> u16 {
        // Flush HWM before entering (for outer scope tracking)
        if self.block_scope_depth == 0 {
            self.pred_hwm = self.pred_hwm.max(self.pred_next);
            self.b32_hwm = self.b32_hwm.max(self.b32_next);
            self.b64_hwm = self.b64_hwm.max(self.b64_next);
            self.f32_hwm = self.f32_hwm.max(self.f32_next);
        }
        self.block_scope_depth += 1;
        // Block-scoped registers start fresh at index 1
        // The outer scope counters are saved on the BlockScopeState in PtxBuilder
        self.block_scope_depth
    }

    /// Exit a native block scope. Returns (pred_count, b32_count, b64_count, f32_count)
    /// for the scope that was just closed.
    fn exit_block_scope(&mut self) -> (u32, u32, u32, u32) {
        let counts = (self.pred_next, self.b32_next, self.b64_next, self.f32_next);
        self.block_scope_depth -= 1;
        counts
    }

    pub fn alloc_pred(&mut self) -> Reg {
        let r = Reg::with_scope(RegClass::Pred, self.pred_next, self.block_scope_depth);
        self.pred_next += 1;
        r
    }

    pub fn alloc_b32(&mut self) -> Reg {
        let r = Reg::with_scope(RegClass::B32, self.b32_next, self.block_scope_depth);
        self.b32_next += 1;
        r
    }

    pub fn alloc_b32_range(&mut self, count: u32) -> Vec<Reg> {
        (0..count).map(|_| self.alloc_b32()).collect()
    }

    pub fn alloc_b64(&mut self) -> Reg {
        let r = Reg::with_scope(RegClass::B64, self.b64_next, self.block_scope_depth);
        self.b64_next += 1;
        r
    }

    pub fn alloc_f32(&mut self) -> Reg {
        let r = Reg::with_scope(RegClass::F32, self.f32_next, self.block_scope_depth);
        self.f32_next += 1;
        r
    }

    pub fn pred_count(&self) -> u32 {
        self.pred_hwm.max(self.pred_next)
    }
    pub fn b32_count(&self) -> u32 {
        self.b32_hwm.max(self.b32_next)
    }
    pub fn b64_count(&self) -> u32 {
        self.b64_hwm.max(self.b64_next)
    }
    pub fn f32_count(&self) -> u32 {
        self.f32_hwm.max(self.f32_next)
    }
}

/// Saved state for a native PTX block scope.
#[derive(Clone, Debug)]
struct BlockScopeState {
    /// Byte offset in `body` where the `{` was emitted. We insert `.reg`
    /// declarations right after this position when `end_scope()` is called.
    body_insert_pos: usize,
    /// Outer scope's register counters (to restore on end_scope).
    pred_next: u32,
    b32_next: u32,
    b64_next: u32,
    f32_next: u32,
    /// The scope_id for registers allocated in this scope.
    scope_id: u16,
}

/// PTX code builder -- emits instructions as formatted strings.
pub struct PtxBuilder {
    pub config: GemmConfig,
    pub regs: RegAllocator,
    pub body: String,
    /// Stack of native block scope states.
    block_scope_stack: Vec<BlockScopeState>,
}

impl PtxBuilder {
    pub fn new(config: GemmConfig) -> Self {
        Self {
            config,
            regs: RegAllocator::new(),
            body: String::with_capacity(32 * 1024),
            block_scope_stack: Vec::new(),
        }
    }

    /// Save register allocation state (legacy soft scoping).
    /// Registers allocated within the scope can be reused after `pop_scope()`.
    pub fn push_scope(&mut self) {
        self.regs.push_scope();
    }

    /// Restore register allocation state (legacy soft scoping).
    pub fn pop_scope(&mut self) {
        self.regs.pop_scope();
    }

    /// Open a new PTX native block scope.
    ///
    /// Emits `{` to the PTX body and switches to a new register allocator
    /// with scope-specific prefixes. Registers allocated after this call
    /// get unique names (e.g., `%t1`, `%td1` for scope 1) and will have
    /// `.reg` declarations emitted inside the block.
    ///
    /// Outer-scope registers remain accessible inside the block.
    /// Block-local registers MUST NOT be referenced after `end_scope()`.
    ///
    /// ptxas sees the `{ .reg ... }` and knows inner registers are dead
    /// at the closing brace, enabling physical register reuse across blocks.
    pub fn begin_scope(&mut self) {
        // Emit the opening brace
        writeln!(self.body, "\t{{").unwrap();
        // Record position where .reg declarations will be inserted
        let insert_pos = self.body.len();
        // Save outer scope counters
        let state = BlockScopeState {
            body_insert_pos: insert_pos,
            pred_next: self.regs.pred_next,
            b32_next: self.regs.b32_next,
            b64_next: self.regs.b64_next,
            f32_next: self.regs.f32_next,
            scope_id: self.regs.enter_block_scope(),
        };
        // Reset counters for the new scope
        self.regs.pred_next = 1;
        self.regs.b32_next = 1;
        self.regs.b64_next = 1;
        self.regs.f32_next = 1;
        self.block_scope_stack.push(state);
    }

    /// Close the current PTX native block scope.
    ///
    /// Inserts `.reg` declarations for block-local registers at the start
    /// of the block and emits `}`. All block-local registers die here.
    pub fn end_scope(&mut self) {
        let state = self
            .block_scope_stack
            .pop()
            .expect("end_scope() called without matching begin_scope()");
        let scope_id = state.scope_id;

        // Get final counts for this scope
        let (pred_count, b32_count, b64_count, f32_count) = self.regs.exit_block_scope();

        // Build .reg declaration string
        let mut reg_decls = String::new();
        if pred_count > 1 {
            let prefix = scope_prefix(scope_id, RegClass::Pred);
            writeln!(reg_decls, "\t.reg .pred \t%{}<{}>;", prefix, pred_count).unwrap();
        }
        if b32_count > 1 {
            let prefix = scope_prefix(scope_id, RegClass::B32);
            writeln!(reg_decls, "\t.reg .b32 \t%{}<{}>;", prefix, b32_count).unwrap();
        }
        if b64_count > 1 {
            let prefix = scope_prefix(scope_id, RegClass::B64);
            writeln!(reg_decls, "\t.reg .b64 \t%{}<{}>;", prefix, b64_count).unwrap();
        }
        if f32_count > 1 {
            let prefix = scope_prefix(scope_id, RegClass::F32);
            writeln!(reg_decls, "\t.reg .f32 \t%{}<{}>;", prefix, f32_count).unwrap();
        }

        // Insert declarations at the saved position (right after `{`)
        if !reg_decls.is_empty() {
            self.body.insert_str(state.body_insert_pos, &reg_decls);
        }

        // Restore outer scope counters
        self.regs.pred_next = state.pred_next;
        self.regs.b32_next = state.b32_next;
        self.regs.b64_next = state.b64_next;
        self.regs.f32_next = state.f32_next;

        // Emit closing brace
        writeln!(self.body, "\t}}").unwrap();
    }

    pub fn w(&mut self, s: &str) {
        writeln!(self.body, "\t{}", s).unwrap();
    }

    // -- Arithmetic --

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

    // -- Packed f16x2 arithmetic --

    /// mul.rn.f16x2 d, a, b; — packed f16 multiply (both halves independently)
    pub fn mul_rn_f16x2(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("mul.rn.f16x2 \t{d}, {a}, {b};"));
    }

    /// Pack a single f16 value into both halves of a b32 register.
    /// cvt.rn.f16.f32 tmp, src_f32; mov.b32 d, {{tmp, tmp}};
    /// Actually simpler: cvt, then shl+or to duplicate.
    pub fn pack_f16x2_from_f32(&mut self, d: Reg, src_f32: Reg, tmp: Reg) {
        // Convert f32 → f16 (low 16 bits of tmp)
        self.cvt_rn_f16_f32(tmp, src_f32);
        // Duplicate: d = tmp | (tmp << 16)
        self.shl_b32(d, tmp, 16);
        self.or_b32(d, d, tmp);
    }

    // -- Float arithmetic (f32) --

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

    // -- Predicate --

    pub fn setp_gt_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("setp.gt.s32 \t{d}, {a}, {b};"));
    }
    pub fn setp_gt_s32_imm(&mut self, d: Reg, a: Reg, imm: i32) {
        self.w(&format!("setp.gt.s32 \t{d}, {a}, {imm};"));
    }
    pub fn setp_lt_s32(&mut self, d: Reg, a: Reg, b: Reg) {
        self.w(&format!("setp.lt.s32 \t{d}, {a}, {b};"));
    }

    // -- Memory --

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
    /// Store two b32 values as a vector pair (st.global.v2.b32).
    /// The two values are stored at consecutive 4-byte addresses starting at [addr+offset].
    pub fn st_global_v2_b32(&mut self, addr: Reg, offset: i32, val0: Reg, val1: Reg) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "st.global.v2.b32 \t{addr_str}, {{{val0}, {val1}}};"
        ));
    }

    // -- Vectorized loads --

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

    // -- Shared memory --

    pub fn st_shared_b32(&mut self, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("st.shared.b32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("st.shared.b32 \t[{addr}+{offset}], {val};"));
        }
    }

    pub fn pred_st_shared_b32(&mut self, pred: Reg, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("@{pred} st.shared.b32 \t[{addr}], {val};"));
        } else {
            self.w(&format!(
                "@{pred} st.shared.b32 \t[{addr}+{offset}], {val};"
            ));
        }
    }

    pub fn ld_shared_b32(&mut self, d: Reg, addr: Reg, offset: i32) {
        if offset == 0 {
            self.w(&format!("ld.shared.b32 \t{d}, [{addr}];"));
        } else {
            self.w(&format!("ld.shared.b32 \t{d}, [{addr}+{offset}];"));
        }
    }

    pub fn ld_shared_v4_b32(&mut self, d: [Reg; 4], addr: Reg, offset: i32) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "ld.shared.v4.b32 \t{{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    pub fn st_shared_v4_b32(&mut self, addr: Reg, offset: i32, val: [Reg; 4]) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "st.shared.v4.b32 \t{addr_str}, {{{}, {}, {}, {}}};",
            val[0], val[1], val[2], val[3]
        ));
    }

    // -- Half-precision conversions --

    pub fn cvt_f32_f16(&mut self, d: Reg, src: Reg) {
        self.w(&format!("cvt.f32.f16 \t{d}, {src};"));
    }

    pub fn cvt_rn_f16_f32(&mut self, d: Reg, src: Reg) {
        self.w(&format!("cvt.rn.f16.f32 \t{d}, {src};"));
    }

    // -- FMA --

    pub fn fma_f32(&mut self, d: Reg, a: Reg, b: Reg, c: Reg) {
        self.w(&format!("fma.rn.f32 \t{d}, {a}, {b}, {c};"));
    }

    // -- rsqrt --

    pub fn rsqrt_approx_f32(&mut self, d: Reg, a: Reg) {
        self.w(&format!("rsqrt.approx.f32 \t{d}, {a};"));
    }

    // -- Shuffle --

    pub fn shfl_bfly(&mut self, d: Reg, src: Reg, offset: u32) {
        self.w(&format!(
            "shfl.sync.bfly.b32 \t{d}, {src}, {offset}, 0x1F, 0xFFFFFFFF;"
        ));
    }

    // -- Predicated operations --

    pub fn pred_bra_neg(&mut self, pred: Reg, label: &str) {
        self.w(&format!("@!{pred} bra \t{label};"));
    }

    pub fn setp_eq_s32(&mut self, d: Reg, a: Reg, imm: i32) {
        self.w(&format!("setp.eq.s32 \t{d}, {a}, {imm};"));
    }

    // -- Async copy --

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

    // -- Tensor core --

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
            d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
    }

    /// MMA with f16 accumulators: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    /// d and c are [Reg; 2] (2 packed f16x2 outputs, not 4 f32).
    pub fn mma_m16n8k16_f16(&mut self, d: [Reg; 2], a: [Reg; 4], b: [Reg; 2], c: [Reg; 2]) {
        self.w(&format!(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \
             {{{},{}}}, {{{},{},{},{}}}, {{{},{}}}, {{{},{}}};",
            d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]
        ));
    }

    /// Emit raw inline PTX (for exp2 blocks, sigmoid, etc.)
    pub fn raw(&mut self, s: &str) {
        writeln!(self.body, "{}", s).unwrap();
    }

    /// st.shared.u32 [addr+offset], val;
    pub fn st_shared_u32(&mut self, addr: Reg, offset: i32, val: Reg) {
        if offset == 0 {
            self.w(&format!("st.shared.u32 \t[{addr}], {val};"));
        } else {
            self.w(&format!("st.shared.u32 \t[{addr}+{offset}], {val};"));
        }
    }

    /// ld.shared.v4.u32 {d0,d1,d2,d3}, [addr+offset];
    pub fn ld_shared_v4_u32(&mut self, d: [Reg; 4], addr: Reg, offset: i32) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "ld.shared.v4.u32 \t{{{}, {}, {}, {}}}, {addr_str};",
            d[0], d[1], d[2], d[3]
        ));
    }

    /// Predicated st.global.v4.u32 [addr+offset], {v0,v1,v2,v3};
    pub fn pred_st_global_v4_u32(
        &mut self,
        pred: Reg,
        addr: Reg,
        offset: i32,
        val: [Reg; 4],
    ) {
        let addr_str = if offset == 0 {
            format!("[{addr}]")
        } else {
            format!("[{addr}+{offset}]")
        };
        self.w(&format!(
            "@{pred} st.global.v4.u32 \t{addr_str}, {{{}, {}, {}, {}}};",
            val[0], val[1], val[2], val[3]
        ));
    }

    // -- Control flow --

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

    // -- Finalize --

    pub fn finalize(&self, kernel_name: &str, params: &[(&str, &str)]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use config::GemmConfig;

    #[test]
    fn test_block_scoping_emits_braces_and_reg_declarations() {
        let config = GemmConfig::default_64x64();
        let mut ptx = PtxBuilder::new(config);

        // Allocate outer-scope registers
        let outer_r1 = ptx.regs.alloc_b32();
        let _outer_rd1 = ptx.regs.alloc_b64();
        ptx.mov_b32_imm(outer_r1, 42);

        // Open a scope
        ptx.begin_scope();
        let inner_r1 = ptx.regs.alloc_b32();
        let inner_f1 = ptx.regs.alloc_f32();
        ptx.mov_b32_imm(inner_r1, 99);
        ptx.mov_f32_imm(inner_f1, 1.0);
        ptx.end_scope();

        // After end_scope, outer counters should be restored
        let outer_r2 = ptx.regs.alloc_b32();
        assert_eq!(outer_r2.scope_id, 0);
        assert_eq!(outer_r2.index, 2); // continues from where outer left off

        // Verify inner registers have scope_id=1
        assert_eq!(inner_r1.scope_id, 1);
        assert_eq!(inner_r1.index, 1);
        assert_eq!(inner_f1.scope_id, 1);

        // Verify the body contains { .reg ... } block
        let body = &ptx.body;
        assert!(body.contains("{"), "Body should contain opening brace");
        assert!(body.contains("}"), "Body should contain closing brace");
        assert!(
            body.contains(".reg .b32 \t%t<2>;"),
            "Body should contain block-local b32 .reg decl, got: {}",
            body
        );
        assert!(
            body.contains(".reg .f32 \t%tf<2>;"),
            "Body should contain block-local f32 .reg decl, got: {}",
            body
        );

        // Verify inner register names use scope prefix
        assert!(
            body.contains("%t1"),
            "Body should reference %t1 (scope 1 b32)"
        );
        assert!(
            body.contains("%tf1"),
            "Body should reference %tf1 (scope 1 f32)"
        );

        // Verify outer registers use standard prefix
        assert!(
            body.contains("%r1"),
            "Body should reference %r1 (outer b32)"
        );
    }

    #[test]
    fn test_multiple_block_scopes_sequential() {
        let config = GemmConfig::default_64x64();
        let mut ptx = PtxBuilder::new(config);

        // Scope 1
        ptx.begin_scope();
        let _s1_r1 = ptx.regs.alloc_b32();
        let _s1_r2 = ptx.regs.alloc_b32();
        ptx.end_scope();

        // Scope 2 (reuses scope_id 1 since we're at the same depth)
        // Actually, scope_id is based on block_scope_depth which goes 0->1->0->1
        ptx.begin_scope();
        let s2_r1 = ptx.regs.alloc_b32();
        ptx.end_scope();

        // Both scopes should have scope_id=1 (same depth)
        assert_eq!(s2_r1.scope_id, 1);
        assert_eq!(s2_r1.index, 1);

        // Body should contain two { } blocks
        let body = &ptx.body;
        let open_count = body.matches("\t{").count();
        let close_count = body.matches("\t}").count();
        assert_eq!(open_count, 2, "Should have 2 opening braces");
        assert_eq!(close_count, 2, "Should have 2 closing braces");
    }

    #[test]
    fn test_outer_scope_reg_counts_unaffected_by_block_scope() {
        let config = GemmConfig::default_64x64();
        let mut ptx = PtxBuilder::new(config);

        // Allocate 5 outer regs
        for _ in 0..5 {
            ptx.regs.alloc_b32();
        }

        // Block scope with 100 inner regs
        ptx.begin_scope();
        for _ in 0..100 {
            ptx.regs.alloc_b32();
        }
        ptx.end_scope();

        // Outer scope should still report 5 (well, 6 because count is next index)
        // The HWM should be max(5+1, 5+1) = 6 since block scope doesn't affect outer HWM
        assert_eq!(ptx.regs.b32_count(), 6, "Outer b32 count should be 6");
    }
}
