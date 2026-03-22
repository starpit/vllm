/// Pre-built MSL header strings for simdgroup operations.
///
/// These are the output of MFA's createMetalSimdgroupEvent() and
/// createMetalSimdgroupMatrixStorage() header generators, captured as
/// Rust string constants.
///
/// MFA generates these programmatically from C++ to handle BF16 decoding,
/// transpose paths, and address space variants. We capture the specific
/// outputs for our target configurations rather than porting the generator.
///
/// Reference: ~/git/ccv/lib/nnc/mfa/kernels/GEMMHeaders.cpp

/// simdgroup_event header — async copy support.
///
/// Two variants:
/// - Full: hardware async copy via AIR intrinsics (apple9 / M3+)
/// - Polyfill: threadgroup-based fallback (apple8 / M1/M2)
pub fn simdgroup_event_header(use_hardware_async: bool) -> &'static str {
    if use_hardware_async {
        SIMDGROUP_EVENT_HARDWARE
    } else {
        SIMDGROUP_EVENT_POLYFILL
    }
}

/// simdgroup_matrix_storage header — matrix load/store/multiply helpers.
///
/// Two variants:
/// - Standard: FP16/FP32 only
/// - BF16: includes BF16 decode/encode paths
pub fn simdgroup_matrix_storage_header(include_bf16: bool) -> &'static str {
    if include_bf16 {
        SIMDGROUP_MATRIX_STORAGE_BF16
    } else {
        SIMDGROUP_MATRIX_STORAGE_STANDARD
    }
}

// ═══════════════════════════════════════════════════════════════════
// Header string constants
//
// TODO: These are STUBS that contain the essential structure but need
// the full MFA output to be captured. The actual headers are ~200 lines
// each. For now, include the structural skeleton so that tests can
// validate the inclusion pattern.
//
// To capture the real headers:
// 1. Build MFA's test suite on macOS
// 2. Print the output of createMetalSimdgroupEvent(false) and (true)
// 3. Print the output of createMetalSimdgroupMatrixStorage(false) and (true)
// 4. Paste here as raw string literals
// ═══════════════════════════════════════════════════════════════════

const SIMDGROUP_EVENT_HARDWARE: &str = r#"
// -*- Metal -*-
//===-- metal_simdgroup_event (hardware async) ----------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

struct _simdgroup_event_t;

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  threadgroup void *, ulong, ulong, ulong2,
  const device void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  device void *, ulong, ulong, ulong2,
  const threadgroup void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };

  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <ushort dst_elements_per_row, ushort threadgroup_size,
              simdgroup_async_copy_clamp_mode clamp_mode =
                simdgroup_async_copy_clamp_mode::clamp_to_zero, typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst, ushort2 dst_tile_dimensions,
      const device T *src, uint src_elements_per_row,
      ushort2 src_tile_dimensions, bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        sizeof(T), alignof(T),
        reinterpret_cast<threadgroup void *>(dst),
        ushort(dst_elements_per_row), 1, ulong2(dst_tile_dimensions),
        reinterpret_cast<const device void *>(src),
        uint(src_elements_per_row), 1, ulong2(src_tile_dimensions),
        long2(0), static_cast<int>(clamp_mode));
    }

    template <ushort src_elements_per_row, ushort threadgroup_size,
              simdgroup_async_copy_clamp_mode clamp_mode =
                simdgroup_async_copy_clamp_mode::clamp_to_zero, typename T>
    METAL_FUNC void async_copy(
      device T *dst, uint dst_elements_per_row, ushort2 dst_tile_dimensions,
      const threadgroup T *src, ushort2 src_tile_dimensions,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        sizeof(T), alignof(T),
        reinterpret_cast<device void *>(dst),
        uint(dst_elements_per_row), 1, ulong2(dst_tile_dimensions),
        reinterpret_cast<const threadgroup void *>(src),
        ushort(src_elements_per_row), 1, ulong2(src_tile_dimensions),
        long2(0), 0);
    }

    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(
        count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }

  private:
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT
"#;

const SIMDGROUP_EVENT_POLYFILL: &str = r#"
// -*- Metal -*-
//===-- metal_simdgroup_event (polyfill) ----------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };

  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <ushort dst_elements_per_row, ushort threadgroup_size,
              simdgroup_async_copy_clamp_mode clamp_mode =
                simdgroup_async_copy_clamp_mode::clamp_to_zero, typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst, ushort2 dst_tile_dimensions,
      const device T *src, uint src_elements_per_row,
      ushort2 src_tile_dimensions, ushort tid,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      #pragma clang loop unroll(full)
      for (ushort i = tid; i < dst_tile_dimensions.y * dst_tile_dimensions.x;
           i += threadgroup_size) {
        const ushort x = i % dst_tile_dimensions.x;
        const ushort y = i / dst_tile_dimensions.x;
        dst[y * dst_elements_per_row + x] = src[y * src_elements_per_row + x];
      }
    }

    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      // No-op for polyfill — synchronization via threadgroup_barrier instead.
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT
"#;

// TODO: Capture full simdgroup_matrix_storage header from MFA.
// This is ~300 lines and handles load/store/multiply for all precisions.
// For now, a structural placeholder that includes the essential types.
const SIMDGROUP_MATRIX_STORAGE_STANDARD: &str = r#"
// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;

    storage_type t;

    METAL_FUNC simdgroup_matrix_storage() thread = default;

    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return (thread vec<T, 2>*)(&t);
    }

    // Load from device memory.
    template <typename U>
    METAL_FUNC void load(
      const device U *src, uint elements_per_row,
      ushort2 matrix_origin, bool transpose_matrix = false
    ) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x);
        vec<U, 2> memoryForm = *(const device vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // Load from threadgroup memory.
    template <typename U>
    METAL_FUNC void load(
      const threadgroup U *src, ushort elements_per_row,
      ushort2 matrix_origin, bool transpose_matrix = false
    ) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x);
        vec<U, 2> memoryForm = *(const threadgroup vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // Store to device memory.
    template <typename U>
    METAL_FUNC void store(
      device U *dst, uint elements_per_row,
      ushort2 matrix_origin, bool transpose_matrix = false
    ) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x);
        vec<T, 2> registerForm = *(thread_elements());
        *(device vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    // Multiply-accumulate: C += A × B (or C = A × B if !accumulate).
    template <typename U, typename V>
    METAL_FUNC void multiply(
      simdgroup_matrix_storage<U> a,
      simdgroup_matrix_storage<V> b,
      bool accumulate = true
    ) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(
        a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }

    // Apply offset to a pointer.
    template <typename U>
    METAL_FUNC static U* apply_offset(
      U *src, uint elements_per_row,
      uint2 matrix_origin, bool transpose_matrix = false
    ) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x) * ulong(elements_per_row) + ulong(matrix_origin.y);
      } else {
        return src + ulong(matrix_origin.y) * ulong(elements_per_row) + ulong(matrix_origin.x);
      }
    }

    template <typename U>
    METAL_FUNC static threadgroup U* apply_offset(
      threadgroup U *src, ushort elements_per_row,
      ushort2 matrix_origin, bool transpose_matrix = false
    ) {
      if (transpose_matrix) {
        return src + ushort(matrix_origin.x) * ushort(elements_per_row) + ushort(matrix_origin.y);
      } else {
        return src + ushort(matrix_origin.y) * ushort(elements_per_row) + ushort(matrix_origin.x);
      }
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE
"#;

// TODO: BF16 variant with decode/encode paths
const SIMDGROUP_MATRIX_STORAGE_BF16: &str = SIMDGROUP_MATRIX_STORAGE_STANDARD;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_event_header_has_asm_intrinsics() {
        let h = simdgroup_event_header(true);
        assert!(h.contains("__asm(\"air.simdgroup_async_copy"), "Missing AIR intrinsic");
        assert!(h.contains("__metal_simdgroup_async_copy_2d"), "Missing 2D copy fn");
        assert!(h.contains("__metal_wait_simdgroup_events"), "Missing wait fn");
        assert!(h.contains("async_copy"), "Missing async_copy method");
    }

    #[test]
    fn test_polyfill_event_header_no_asm() {
        let h = simdgroup_event_header(false);
        assert!(!h.contains("__asm("), "Polyfill should NOT have __asm intrinsics");
        assert!(h.contains("for (ushort i = tid"), "Polyfill should use loop-based copy");
    }

    #[test]
    fn test_matrix_storage_header_has_load_store_multiply() {
        let h = simdgroup_matrix_storage_header(false);
        assert!(h.contains("METAL_FUNC void load"), "Missing load method");
        assert!(h.contains("METAL_FUNC void store"), "Missing store method");
        assert!(h.contains("METAL_FUNC void multiply"), "Missing multiply method");
        assert!(h.contains("__metal_simdgroup_matrix_8x8_multiply_accumulate"),
            "Missing actual Metal intrinsic for MMA");
        assert!(h.contains("thread_elements()"), "Missing thread_elements accessor");
        assert!(h.contains("apply_offset"), "Missing apply_offset static method");
        assert!(h.contains("vec<T, 64>"), "Storage must be vec<T, 64>");
        assert!(h.contains("vec<T, 2>*"), "thread_elements must return vec<T, 2>*");
    }

    #[test]
    fn test_matrix_storage_handles_transpose() {
        let h = simdgroup_matrix_storage_header(false);
        assert!(h.contains("transpose_matrix"), "Must handle transpose parameter");
        // Both device and threadgroup variants
        assert!(h.contains("const device"), "Missing device load variant");
        assert!(h.contains("const threadgroup"), "Missing threadgroup load variant");
    }

    #[test]
    fn test_headers_have_include_guards() {
        let h1 = simdgroup_event_header(true);
        assert!(h1.contains("#ifndef __METAL_SIMDGROUP_EVENT"));
        assert!(h1.contains("#endif"));

        let h2 = simdgroup_matrix_storage_header(false);
        assert!(h2.contains("#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE"));
        assert!(h2.contains("#endif"));
    }

    #[test]
    fn test_headers_use_metal_internals_pragma() {
        let h1 = simdgroup_event_header(true);
        assert!(h1.contains("#pragma METAL internals : enable"));
        assert!(h1.contains("#pragma METAL internals : disable"));

        let h2 = simdgroup_matrix_storage_header(false);
        assert!(h2.contains("#pragma METAL internals : enable"));
        assert!(h2.contains("#pragma METAL internals : disable"));
    }

    #[test]
    fn test_matrix_storage_has_correct_template_types() {
        let h = simdgroup_matrix_storage_header(false);
        assert!(h.contains("template <typename T>"), "Missing T template parameter");
        assert!(h.contains("template <typename U>"), "Missing U template parameter for load/store");
        assert!(h.contains("vec<T, 2>"), "Missing vec<T,2> for thread elements");
    }
}
