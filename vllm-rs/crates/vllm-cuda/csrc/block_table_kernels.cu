// SPDX-License-Identifier: Apache-2.0
// Block table gather kernel — GPU-side reorder from persistent to input table.
//
// Matches Python vLLM's _gather_block_tables_kernel: copies only the used
// blocks per row, reordering by batch index → req index mapping.

#include <cstdint>

// One thread-block per batch row. Vectorized 128-bit (int4) loads where possible.
__global__ void gather_block_table_kernel(
    const int32_t* __restrict__ src,         // [max_reqs, src_stride]
    int32_t* __restrict__ dst,               // [batch_size, dst_stride]
    const int32_t* __restrict__ num_blocks,  // [max_reqs]
    const int32_t* __restrict__ req_indices, // [batch_size]
    int src_stride,
    int dst_stride
) {
    const int batch_idx = blockIdx.x;
    const int req_idx = req_indices[batch_idx];
    const int n = num_blocks[req_idx];

    const int32_t* src_row = src + req_idx * src_stride;
    int32_t* dst_row = dst + batch_idx * dst_stride;

    // Vectorized copy: 4 ints at a time (128 bits).
    const int n4 = n / 4;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int4* src4 = reinterpret_cast<const int4*>(src_row);
    int4* dst4 = reinterpret_cast<int4*>(dst_row);
    for (int i = tid; i < n4; i += stride) {
        dst4[i] = src4[i];
    }

    // Remainder (0-3 elements).
    const int base = n4 * 4;
    for (int i = base + tid; i < n; i += stride) {
        dst_row[i] = src_row[i];
    }
}

extern "C" {

void gather_block_table(
    const void* src,
    void* dst,
    const void* num_blocks,
    const void* req_indices,
    int src_stride,
    int dst_stride,
    int batch_size,
    void* stream
) {
    if (batch_size == 0) return;
    // 256 threads per block — enough for vectorized copy of typical block counts.
    const int threads = 256;
    gather_block_table_kernel<<<batch_size, threads, 0, (cudaStream_t)stream>>>(
        (const int32_t*)src,
        (int32_t*)dst,
        (const int32_t*)num_blocks,
        (const int32_t*)req_indices,
        src_stride,
        dst_stride
    );
}

} // extern "C"
