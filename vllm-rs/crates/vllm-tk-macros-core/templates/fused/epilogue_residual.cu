        {   rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> acc_bf;
            warp::copy(acc_bf, acc);
            rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> res_bf;
            warp::load(res_bf, {{ residual }}, {{"{"}}{{ row_var }}, col});
            #pragma unroll
            for (int r = 0; r < acc_bf.height; r++)
                #pragma unroll
                for (int c = 0; c < acc_bf.width; c++)
                    #pragma unroll
                    for (int k = 0; k < acc_bf.tiles[0][0].packed_per_thread; k++) {
                        bf16_2 &a = acc_bf.tiles[r][c].data[k];
                        bf16_2 &rv = res_bf.tiles[r][c].data[k];
                        float a_lo = __bfloat162float(__low2bfloat16(a));
                        float a_hi = __bfloat162float(__high2bfloat16(a));
                        float r_lo = __bfloat162float(__low2bfloat16(rv));
                        float r_hi = __bfloat162float(__high2bfloat16(rv));
                        a = __floats2bfloat162_rn(a_lo + r_lo, a_hi + r_hi);
                    }
            warp::store({{ residual }}, acc_bf, {{"{"}}{{ row_var }}, col});
        }
