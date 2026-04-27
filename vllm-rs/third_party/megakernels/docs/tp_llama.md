# High-Throughput Multi-GPU Llama Overview

## Overview

### Instruction Set 

This megakernel implements Llama-70B with tensor parallelism sharded across eight H100 GPUs. We've broken down the model into the following instruction set:

- Attn RMS norm 
- QKV + rope + KV cache append
- Attention Decode
- Oproj + residual connection
- MLP RMS norm
- Gate MLP layer + silu
- UP MLP layer + elementwise multiplication with gate output
- Down MLP layer + residual connection
- LM head RMS norm
- LM head

The RMS norm is batched so that a single instruction can handle multiple vectors. Similarly, the attention decode is batched so that a single instruction can handle eight (batch index, kv head index) pairs. Also, oproj and downproj use the same code backbone (in matmul_adds).

### Parallelism Structure

Our parallelism structure is as follows:

At the start of each layer going into the attention norm operation, the Activations are sharded at the batch level, so GPU0 has the first eighth of tokens, GPU1 has the next eighth, etc. The attention-norm operation is computed with this data parallel structure. However, at the end of each norm operation, the data is broadcasted from each GPU to the rest of the other GPUs. Therefore, at the beginning of the QKV operation, each GPU has a full copy of the post-norm data.

The QKV projection is tensor parallel, so each GPU has a slice of the projection matrix and therefore generates one KV head and the corresponding Q heads (the Llama model uses GQA).

Attention is also completed tensor parallel, so each GPU processes all of the tokens but for only a slice of the corresponding attention heads.

At the end of attention, a relatively custom "distributed transpose" operation happens, where we repartition the data from being tensor parallel to being data parallel. So at the end of this distributed transpose operation, each GPU once again has the full activation vectors for a slice of the total tokens in the batch.

The output projection and residual connection happens purely data parallel. This is the only linear layer that is not sharded across the GPUs and is instead replicated on each GPU.

Similar to the attention norm, the MLP norm is computed purely data-parallel and also contains a broadcast operation at the end so that all GPUs contain the full activation vectors for all the tokens after this operation.

The MLP is computed with tensor parallelism, with the sharding happening across the intermediate dimension of the MLP. The final down projection MLP layer also implements a reduced scatter, so that at the end of the down projection, each GPU has only a slice of the data, but it's fully aggregated from across the other GPUs.


## Relevant Files

The CUDA source is in `demo/cross-gpu-llama`. On the Python side, the primary entry point for running things is `megakernels/scripts/tp_generate.py`. Many of the imported files from that script are in `megakernels/demos/tp_throughput`, particularly `instructions.py` and `scheduler.py`.

## Running Things

To compile the kernel, `cd` into `demo/cross-gpu-llama` and run `make`.

To run the kernel use (for example):

```bash

python megakernels/scripts/tp_generate.py skip_weight_load=F do_benchmark=T chat=T global_work_queue=T interleave_waves=T glob_bs=4096 ntok=4

```