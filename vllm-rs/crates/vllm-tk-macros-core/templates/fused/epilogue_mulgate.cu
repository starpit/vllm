        {   rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> acc_bf;
            warp::copy(acc_bf, acc);
            rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> gate_bf;
            warp::load(gate_bf, {{ gate_output }}, {{"{"}}{{ row_var }}, col});
            #pragma unroll
            for (int r = 0; r < acc_bf.height; r++)
                #pragma unroll
                for (int c = 0; c < acc_bf.width; c++)
                    #pragma unroll
                    for (int k = 0; k < acc_bf.tiles[0][0].packed_per_thread; k++) {
                        bf16_2 &a = acc_bf.tiles[r][c].data[k];
                        bf16_2 &gv = gate_bf.tiles[r][c].data[k];
                        float a_lo = __bfloat162float(__low2bfloat16(a));
                        float a_hi = __bfloat162float(__high2bfloat16(a));
                        float g_lo = __bfloat162float(__low2bfloat16(gv));
                        float g_hi = __bfloat162float(__high2bfloat16(gv));
                        a = __floats2bfloat162_rn(a_lo * g_lo, a_hi * g_hi);
                    }
            warp::store({{ gate_output }}, acc_bf, {{"{"}}{{ row_var }}, col});
        }
