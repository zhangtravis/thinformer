# LongBench Language Modeling

## Prerequisites

## Dependencies

```bash
# Create conda environment
conda create -n thinformer-longbench python=3.12
conda activate thinformer-longbench
pip install torch==2.6 torchvision torchaudio
pip install flash-attn --no-build-isolation
# Install other needed Python packages
pip install tqdm datasets transformers==4.40.2 sentencepiece
# Install thinformer
cd ../..
pip install -e .
```

> \[!WARNING\]
> The original instructions for installing triton using `pip install triton==2.0.0.dev20221202 --no-deps` from the `hyper-attn` repo is outdated. If installing `torch==2.6.0`, triton 3.2.0 will automatically be installed.

## Results

Note: all results were run using an Nvidia A6000 GPU with 48 GB memory with CUDA 12.1.

### Runtime

**Exact:**

```bash
python benchmark_single_attention.py --attn_method flash-cuda --mode fwd
```

Expected output:
```bash
no_causal        : False
mode             : fwd+bwd
attn_method      : flash-cuda
mode: fwd+bwd, attn_method: flash-cuda, batch_size: 1, head_size: 32, dim: 64
[fwd     ], flash-cuda, seq_len: 1024    , causal: True, ms: 0.09421 (0.09728, 0.10035) |
[fwd     ], flash-cuda, seq_len: 2048    , causal: True, ms: 0.27034 (0.27546, 0.28058) |
[fwd     ], flash-cuda, seq_len: 4096    , causal: True, ms: 0.86528 (0.87552, 0.88781) |
[fwd     ], flash-cuda, seq_len: 8192    , causal: True, ms: 3.08859 (3.11194, 3.13631) |
[fwd     ], flash-cuda, seq_len: 16384   , causal: True, ms: 12.01234 (12.08730, 12.15857) |
[fwd     ], flash-cuda, seq_len: 32768   , causal: True, ms: 51.96738 (52.15846, 52.34954) |
[fwd     ], flash-cuda, seq_len: 65536   , causal: True, ms: 198.91609 (198.91609, 198.91609) |
[fwd     ], flash-cuda, seq_len: 131072  , causal: True, ms: 841.84473 (841.84473, 841.84473) |
```

**HyperAttention:**

```bash
python benchmark_single_attention.py --attn_method hyper-cuda --mode fwd
```

```bash
[fwd     ], hyper-cuda, seq_len: 1024    , causal: True, ms: 0.09318 (0.09523, 0.09933) |
[fwd     ], hyper-cuda, seq_len: 2048    , causal: True, ms: 0.26726 (0.27136, 0.27750) |
[fwd     ], hyper-cuda, seq_len: 4096    , causal: True, ms: 0.87020 (0.88474, 0.89416) |
[fwd     ], hyper-cuda, seq_len: 8192    , causal: True, ms: 3.19959 (3.21843, 3.22744) |
[fwd     ], hyper-cuda, seq_len: 16384   , causal: True, ms: 9.22993 (9.23853, 9.25348) |
[fwd     ], hyper-cuda, seq_len: 32768   , causal: True, ms: 24.16988 (24.24013, 24.28498) |
[fwd     ], hyper-cuda, seq_len: 65536   , causal: True, ms: 60.17126 (60.17126, 60.17126) |
[fwd     ], hyper-cuda, seq_len: 131072  , causal: True, ms: 145.29126 (145.29126, 145.29126) |
```

**Thinformer:**

```bash
python benchmark_single_attention.py --attn_method thinformer --mode fwd
```

```bash
[fwd     ], thinformer, seq_len: 1024    , causal: True, ms: 0.09318 (0.09626, 0.09933) |
[fwd     ], thinformer, seq_len: 2048    , causal: True, ms: 0.26481 (0.27034, 0.27689) |
[fwd     ], thinformer, seq_len: 4096    , causal: True, ms: 0.85709 (0.87347, 0.88474) |
[fwd     ], thinformer, seq_len: 8192    , causal: True, ms: 38.07109 (38.12147, 38.17185) |
[fwd     ], thinformer, seq_len: 16384   , causal: True, ms: 76.75290 (76.75290, 76.75290) |
[fwd     ], thinformer, seq_len: 32768   , causal: True, ms: 153.73619 (153.73619, 153.73619) |
[fwd     ], thinformer, seq_len: 65536   , causal: True, ms: 244.14310 (244.14310, 244.14310) |
[fwd     ], thinformer, seq_len: 131072  , causal: True, ms: 484.33151 (484.33151, 484.33151) |
```

### Perplexity

**Exact:**

```bash
python benchmark_patch_llm.py --attn_method flash --seq_len 32768
```

Expected result:

```bash
ppl: 5.634139089948601, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 144, ppl: 5.634139089948601, nan_cnt: 0
```

**HyperAttention:**

```bash
python benchmark_patch_llm.py --attn_method hyper-cuda --seq_len 32768
```

Expected result:

```bash
ppl: 13.547103060846743, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 115, ppl: 13.547103060846743, nan_cnt: 0
```

**Thinformer:**

```bash
python benchmark_patch_llm.py --attn_method thinformer --seq_len 32768
```

Expected result:

```bash
ppl: 16.014302979344908, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 115, ppl: 16.014302979344908, nan_cnt: 0
```

### Perplexity over all datasets
**Exact:**

```bash
python benchmark_patch_llm.py --attn_method flash --seq_len 32768
```

Expected result:

```bash
seq_len=1024
ppl: 9.626026316439225, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 1024, num_patch_layers: -1, n_data: 4723, ppl: 9.626026316439225, nan_cnt: 0
```

```bash
seq_len=2048
ppl: 8.949646466523212, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 2048, num_patch_layers: -1, n_data: 4558, ppl: 8.949646466523212, nan_cnt: 0
```

```bash
seq_len=4096
ppl: 8.71016506767979, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 4096, num_patch_layers: -1, n_data: 3782, ppl: 8.71016506767979, nan_cnt: 0
```

```bash
seq_len=8192
ppl: 7.911487140205213, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 8192, num_patch_layers: -1, n_data: 2552, ppl: 7.911487140205213, nan_cnt: 0
```

```bash
seq_len=16384
ppl: 6.244348367769085, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 16384, num_patch_layers: -1, n_data: 1002, ppl: 6.244348367769085,nan_cnt: 0
```


```bash
seq_len=32768
ppl: 5.634139089948601, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 144, ppl: 5.634139089948601, nan_cnt: 0
```

**HyperAttention:**

```bash
python benchmark_patch_llm.py --attn_method hyper-cuda --seq_len 32768
```

Expected result:

```bash
seq_len=1024
ppl: 9.625692018381297, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 1024, num_patch_layers: -1, n_data: 4723, ppl: 9.625692018381297, nan_cnt: 0
```

```bash
seq_len=2048
ppl: 8.948984028818106, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 2048, num_patch_layers: -1, n_data: 4558, ppl: 8.948984028818106, nan_cnt: 0
```

```bash
seq_len=4096
ppl: 8.709957130020854, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 4096, num_patch_layers: -1, n_data: 3782, ppl: 8.709957130020854, nan_cnt: 0
```

```bash
seq_len=8192
ppl: 11.74498107884558, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 8192, num_patch_layers: -1, n_data: 2552, ppl: 11.74498107884558, nan_cnt: 0
```

```bash
seq_len=16384
ppl: 12.293787711513733, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 16384, num_patch_layers: -1, n_data: 1002, ppl: 12.293787711513733, nan_cnt: 0
```

```bash
seq_len=32768
ppl: 12.15541132622295, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 144, ppl: 12.15541132622295, nan_cnt: 0
```

**Thinformer:**

```bash
python benchmark_patch_llm.py --attn_method thinformer --seq_len 32768
```

Expected result:

```bash
seq_len=1024
ppl: 9.625692018381297, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 1024, num_patch_layers: -1, n_data: 4723, ppl: 9.625692018381297, nan_cnt: 0
```

```bash
seq_len=2048
ppl: 8.948984028818106, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 2048, num_patch_layers: -1, n_data: 4558, ppl: 8.948984028818106, nan_cnt: 0
```

```bash
seq_len=4096
ppl: 8.709957130020854, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 4096, num_patch_layers: -1, n_data: 3782, ppl: 8.709957130020854, nan_cnt: 0
```


```bash
seq_len=8192
ppl: 11.847297953307441, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 8192, num_patch_layers: -1, n_data: 2552, ppl: 11.847297953307441,nan_cnt: 0
```

```bash
seq_len=16384
ppl: 10.985162452547375, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 16384, num_patch_layers: -1, n_data: 1002, ppl: 10.985162452547375, nan_cnt: 0
```

```bash
seq_len=32768
ppl: 13.978334309326279, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 144, ppl: 13.978334309326279,nan_cnt: 0
```
