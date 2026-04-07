        {   rt_bf<16, PFL_OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);
            warp::store({{ output }}, out_bf, {{"{"}}{{ row_var }}, col});
        }
