        {   rt_bf<16, PFL_OUT_BLOCK> out_bf;
            #pragma unroll
            for (int i = 0; i < acc.height; i++)
                #pragma unroll
                for (int j = 0; j < acc.width; j++)
                    #pragma unroll
                    for (int d = 0; d < acc.tiles[i][j].num_elements; d++) {
                        float2 &v = acc.tiles[i][j].data[d];
                        v.x = v.x / (1.f + expf(-v.x));
                        v.y = v.y / (1.f + expf(-v.y));
                    }
            warp::copy(out_bf, acc);
            warp::store({{ output }}, out_bf, {{"{"}}{{ row_var }}, col});
        }
