//! PTX kernel fusion: analyze and rewrite PTX kernels at compile time.
//!
//! This crate provides:
//! - `analyze_kernel!("path.ptx")` — extract a kernel's protocol (registers, SMEM, params, I/O)
//! - `rewrite_kernel!("path.ptx", { "%r3" => "%r30" })` — rename registers and re-extract protocol
//!
//! The goal: prove we can automatically determine a kernel's "surface area" from its PTX,
//! then transform the PTX while preserving correctness. This is the foundation for
//! proc-macro-driven kernel fusion.

pub use ptx_fusion_macros::{
    analyze_kernel, analyze_kernel_as, fuse_kernels, regfuse_kernels, rewrite_kernel,
};

/// A kernel's complete protocol — everything needed to fuse it with another kernel.
#[derive(Debug)]
pub struct KernelProtocol {
    pub name: &'static str,
    pub registers: &'static [(&'static str, usize)],
    pub smem_regions: &'static [SmemRegion],
    pub total_smem_bytes: u32,
    pub params: &'static [KernelParam],
    pub global_loads: &'static [DataPort],
    pub global_stores: &'static [DataPort],
    pub smem_loads: u32,
    pub smem_stores: u32,
    pub barriers: &'static [u32],
    pub has_mma: bool,
}

#[derive(Debug)]
pub struct SmemRegion {
    pub name: &'static str,
    pub align: usize,
    pub elem_type: &'static str,
    pub count: usize,
    pub size_bytes: usize,
}

#[derive(Debug)]
pub struct KernelParam {
    pub name: &'static str,
    pub ptx_type: &'static str,
    pub is_pointer: bool,
    pub index: u32,
}

#[derive(Debug)]
pub struct DataPort {
    pub param_name: &'static str,
    pub data_type: &'static str,
    pub line: u32,
}

impl KernelProtocol {
    /// Pretty-print the protocol for inspection.
    pub fn display(&self) {
        println!("╔══ Kernel Protocol: {} ══", self.name);
        println!("║");

        println!("║ Registers:");
        for (ty, count) in self.registers {
            println!("║   {ty:6} × {count}");
        }

        let total_regs: usize = self.registers.iter().map(|(_, c)| c).sum();
        println!("║   ─────────────");
        println!("║   total: {total_regs}");
        println!("║");

        if !self.smem_regions.is_empty() {
            println!("║ Shared Memory: {} bytes total", self.total_smem_bytes);
            for r in self.smem_regions {
                println!(
                    "║   {} — {} × {} ({} bytes, align {})",
                    r.name, r.elem_type, r.count, r.size_bytes, r.align
                );
            }
            println!("║");
        } else {
            println!("║ Shared Memory: none");
            println!("║");
        }

        println!("║ Parameters:");
        for p in self.params {
            let ptr_tag = if p.is_pointer { " [PTR]" } else { "" };
            println!("║   [{}] {} : {}{}", p.index, p.name, p.ptx_type, ptr_tag);
        }
        println!("║");

        println!("║ Global Loads ({}):", self.global_loads.len());
        for d in self.global_loads {
            println!(
                "║   line {:3}: ld.global.{} ← param \"{}\"",
                d.line, d.data_type, d.param_name
            );
        }
        println!("║");

        println!("║ Global Stores ({}):", self.global_stores.len());
        for d in self.global_stores {
            println!(
                "║   line {:3}: st.global.{} → param \"{}\"",
                d.line, d.data_type, d.param_name
            );
        }
        println!("║");

        println!(
            "║ SMEM loads: {}, SMEM stores: {}",
            self.smem_loads, self.smem_stores
        );
        println!("║ Barriers: {:?}", self.barriers);
        println!("║ MMA instructions: {}", self.has_mma);
        println!("╚═══════════════════════════════════");
    }
}
