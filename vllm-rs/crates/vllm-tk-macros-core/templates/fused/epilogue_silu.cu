        {   rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> out_bf;
            #pragma unroll
            for (int i = 0; i < acc.height; i++)
                #pragma unroll
                for (int j = 0; j < acc.width; j++)
                    #pragma unroll
                    // NOTE: use packed_per_thread (= 4), NOT num_elements (= 256
                    // = rows*cols per sub-tile). data[] is a per-thread array
                    // of packed_per_thread entries; num_elements counts the
                    // whole-subtile elements which overshoots by 64x and
                    // reads/writes OOB into neighboring registers.
                    for (int d = 0; d < acc.tiles[0][0].packed_per_thread; d++) {
                        float2 &v = acc.tiles[i][j].data[d];
                        v.x = v.x / (1.f + expf(-v.x));
                        v.y = v.y / (1.f + expf(-v.y));
                    }
            warp::copy(out_bf, acc);
            warp::store({{ output }}, out_bf, {{"{"}}{{ row_var }}, col});
        }
