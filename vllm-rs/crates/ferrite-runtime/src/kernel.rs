use std::ffi::CString;
use std::sync::OnceLock;
use cudarc::driver::result as cuda;
use cudarc::driver::sys::{CUfunction, CUmodule};

/// A JIT-compiled CUDA kernel backed by a PTX string.
///
/// The PTX is compiled lazily on first launch via `OnceLock`.
/// After that, launches reuse the compiled module and function handle.
pub struct JitKernel {
    ptx: &'static str,
    kernel_name: &'static str,
    compiled: OnceLock<CompiledKernel>,
}

struct CompiledKernel {
    _module: CUmodule,
    function: CUfunction,
}

// SAFETY: CUmodule and CUfunction are opaque pointers that can be sent across threads.
// The CUDA driver API is thread-safe for these operations.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl JitKernel {
    /// Create a new JitKernel from a static PTX string and kernel entry name.
    pub const fn new(ptx: &'static str, kernel_name: &'static str) -> Self {
        Self {
            ptx,
            kernel_name,
            compiled: OnceLock::new(),
        }
    }

    /// Lazily compile and return the CUDA function handle.
    fn get_function(&self) -> CUfunction {
        let compiled = self.compiled.get_or_init(|| {
            let ptx_cstr = CString::new(self.ptx).expect("PTX contains null byte");
            let module = unsafe {
                cuda::module::load_data(ptx_cstr.as_ptr() as *const _)
                    .expect("failed to load PTX module")
            };
            let name_cstr = CString::new(self.kernel_name).expect("kernel name contains null byte");
            let function = unsafe {
                cuda::module::get_function(module, name_cstr)
                    .expect("failed to get kernel function")
            };
            CompiledKernel {
                _module: module,
                function,
            }
        });
        compiled.function
    }

    /// Launch the kernel with the given grid, block dimensions, shared memory, and arguments.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - All device pointers in `args` are valid
    /// - Grid and block dimensions are appropriate for the kernel
    /// - A CUDA context is active on the current thread
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &[*mut std::ffi::c_void],
    ) -> Result<(), cudarc::driver::DriverError> {
        let f = self.get_function();
        let mut args_mut: Vec<*mut std::ffi::c_void> = args.to_vec();
        unsafe { cuda::launch_kernel(
            f,
            grid,
            block,
            shared_mem_bytes,
            cuda::stream::null(),
            &mut args_mut,
        ) }
    }
}
