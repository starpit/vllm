// Copyright © 2025 Apple Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NAX (Apple9 / M4+) MMA tile infrastructure.
//
// Vendored from mlx/backend/metal/kernels/steel/gemm/nax.h and the
// supporting steel utilities (defines.h, integral_constant.h, type_traits.h).
// Kept as a stand-alone header so `quantized_qmm_nax.metal` can include it
// without vendoring the full MLX steel tree.
//
// Requires: MetalPerformancePrimitives/MetalPerformancePrimitives.h (included
// below), metal_simdgroup, metal_stdlib.

#pragma once

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_simdgroup>
#include <metal_stdlib>

#pragma METAL internals : enable

// ─────────────────────────────────────────────────────────────────────────────
// Minimal steel utilities (defines.h + integral_constant.h + type_traits.h)
// ─────────────────────────────────────────────────────────────────────────────

#define STEEL_CONST  static constant constexpr const
#define STEEL_FUNC   METAL_FUNC
#define STEEL_PRAGMA_UNROLL    _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

namespace metal {

template <typename T>
struct pointer_element {};
template <typename T> struct pointer_element<thread     T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<device     T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<constant   T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<threadgroup T*> { using type = remove_cv_t<T>; };
template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

namespace mlx {
namespace steel {

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  METAL_FUNC constexpr operator value_type() const noexcept { return value; }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

template <int val>
using Int = integral_constant<int, val>;

template <typename T, T tv, typename U, U uv>
METAL_FUNC constexpr auto operator+(integral_constant<T,tv>, integral_constant<U,uv>) {
  constexpr auto r = tv + uv; return integral_constant<decltype(r), r>{};
}
template <typename T, T tv, typename U, U uv>
METAL_FUNC constexpr auto operator*(integral_constant<T,tv>, integral_constant<U,uv>) {
  constexpr auto r = tv * uv; return integral_constant<decltype(r), r>{};
}

template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) f(true_type{});
  else   f(false_type{});
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    f(Int<start>{});
    const_for_loop<start + step, stop, step, F>(f);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// BaseNAXFrag — vendored from mlx/backend/metal/kernels/steel/gemm/nax.h
// ─────────────────────────────────────────────────────────────────────────────

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;
  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;  // 8

  STEEL_CONST short kElemRows    = 2;
  STEEL_CONST short kElemCols    = 4;
  STEEL_CONST short kElemRowsJump = 8;

  static_assert(kElemRows * kElemCols == kElemsPerFrag, "");

  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort lid = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = lid >> 2;
    const short fm = ((qid & 4) | ((lid >> 1) & 3));
    const short fn = ((qid & 2) | (lid & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort lid = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = lid >> 2;
    const short fm = ((qid & 4) | ((lid >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (lid & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++)
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++)
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
      }
    }
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY = Int<0>,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y = {},
      OffX off_x = {}, OffY off_y = {}) {
    (void)lim_y;  // row-only bounds check; col safety not needed for aligned K tiles
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++)
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++)
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) dst[i * kElemCols + j] = T(0);
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++)
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++)
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y,
      OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly))
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  // MMA: TN%2==0 path — C[mm,nn], C[mm,nn+1] += A[mm,kk] x B[kk,nn,tb]^(tb)
  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), CType>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = A[i];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { ct_b[i] = Bn0[i]; ct_b[kElemsPerFrag + i] = Bn1[i]; }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { ct_c[i] = Cn0[i]; ct_c[kElemsPerFrag + i] = Cn1[i]; }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { Cn0[i] = ct_c[i]; Cn1[i] = ct_c[kElemsPerFrag + i]; }
  }

  // MMA: TN==1 or TM%2==0 path — C[mm,nn] += A^(ta)[mm,kk] x B[kk,nn]^(tb)
  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), CType>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { ct_a[i] = Am0[i]; ct_a[kElemsPerFrag + i] = Am1[i]; }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) ct_b[i] = B[i];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { ct_c[i] = Cm0[i]; ct_c[kElemsPerFrag + i] = Cm1[i]; }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) { Cm0[i] = ct_c[i]; Cm1[i] = ct_c[kElemsPerFrag + i]; }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// NAXTile — vendored from nax.h
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, short kTileRows_, short kTileCols_,
          class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t   = NAXFrag_;
  using elem_type   = T;

  STEEL_CONST short kFragRows    = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols    = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags    = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows  = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols  = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) val_frags[i] = frag_type(0);
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(const short i, const short j) const {
    return val_frags[i * kTileCols + j];
  }
  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() { return val_frags[i * kTileCols + j]; }
  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const { return val_frags[i * kTileCols + j]; }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) {
    if constexpr (transpose) return frag_at(j, i); else return frag_at(i, j);
  }
  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) const {
    if constexpr (transpose) return frag_at(j, i); else return frag_at(i, j);
  }
  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    if constexpr (transpose) return frag_at<j, i>(); else return frag_at<i, j>();
  }
  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    if constexpr (transpose) return frag_at<j, i>(); else return frag_at<i, j>();
  }

  // Load from device (stride = ld)
  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(frag_at<idx_row.value, idx_col.value>(), src, ld, Int<1>{},
                        idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  // Load from device with row-limit guard (M-tail)
  template <typename U>
  METAL_FUNC void load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(frag_at<idx_row.value, idx_col.value>(), src, ld, Int<1>{},
                             src_tile_dims.y, src_tile_dims.x,
                             idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  // Load from threadgroup (compile-time strides)
  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(frag_at<idx_row.value, idx_col.value>(), src,
                        Int<str_x>{}, Int<str_y>{},
                        idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  // Store to device
  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(frag_at<idx_row.value, idx_col.value>(), dst, ld, Int<1>{},
                         idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  // Store to device with (rows, cols) guard
  template <typename U>
  METAL_FUNC void store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(frag_at<idx_row.value, idx_col.value>(), dst, ld, Int<1>{},
                              dst_tile_dims.y, dst_tile_dims.x,
                              idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// tile_matmad_nax — vendored from nax.h
// ─────────────────────────────────────────────────────────────────────────────

template <class CTile, class ATile, class BTile, bool transpose_a, bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C, thread ATile& A, metal::bool_constant<transpose_a>,
    thread BTile& B, metal::bool_constant<transpose_b>) {
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM  = CTile::kTileRows;
  static_assert(TMa == TM, "");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN  = CTile::kTileCols;
  static_assert(TNb == TN, "");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK  = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn), C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta), A.frag_at(mm + 1, kk, ta), ta,
              B.frag_at(kk, nn, tb), tb);
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn), C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta), ta,
              B.frag_at(kk, nn, tb), B.frag_at(kk, nn + 1, tb), tb);
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

// Note: #pragma METAL internals deliberately NOT disabled here.
// Any .metal file that includes metal_nax.h is an M4+-only NAX
// kernel that requires MPP internals throughout (including at
// template-instantiation sites in the including file).
