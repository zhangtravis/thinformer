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
```

> \[!NOTE\]
> The original instructions for installing triton using `pip install triton==2.0.0.dev20221202 --no-deps` from the `hyper-attn` repo is outdated. If installing `torch==2.6.0`, triton 3.2.0 will automatically be installed.

## Results

### Runtime

For the exact attention method, please run:

```bash
python benchmark_single_attention.py --attn_method flash-cuda --mode fwd
```

Expected output:
```bash
no_causal        : False
mode             : fwd+bwd
attn_method      : flash-cuda
mode: fwd+bwd, attn_method: flash-cuda, batch_size: 1, head_size: 32, dim: 64
[fwd+bwd ], flash-cuda, seq_len: 1024    , causal: True, ms: 0.35123 (0.35430, 0.35738) | 
[fwd+bwd ], flash-cuda, seq_len: 2048    , causal: True, ms: 0.91443 (0.92211, 0.92877) | 
[fwd+bwd ], flash-cuda, seq_len: 4096    , causal: True, ms: 2.86802 (2.87795, 2.89567) | 
[fwd+bwd ], flash-cuda, seq_len: 8192    , causal: True, ms: 9.99711 (10.02138, 10.03868) | 
[fwd+bwd ], flash-cuda, seq_len: 16384   , causal: True, ms: 37.70122 (37.95558, 38.11942) | 
[fwd+bwd ], flash-cuda, seq_len: 32768   , causal: True, ms: 147.96513 (148.00538, 148.04562) | 
[fwd+bwd ], flash-cuda, seq_len: 65536   , causal: True, ms: 594.00497 (594.00497, 594.00497) | 
[fwd+bwd ], flash-cuda, seq_len: 131072  , causal: True, ms: 2405.08624 (2405.08624, 2405.08624) |
```

For the thinformer method, please run:

```bash
python benchmark_single_attention.py --attn_method thinformer --mode fwd
```

### Perplexity

For the exact attention method, please run:

```bash
python benchmark_patch_llm.py --attn_method flash --seq_len 32768
```

Expected result:

```bash
ppl: 5.634139089948601, nan_cnt: 0
model: chatglm2-6b-32k, dtype: torch.bfloat16, seq_len: 32768, num_patch_layers: -1, n_data: 144, ppl: 5.634139089948601, nan_cnt: 0
```

For the thinformer method, please run:

```bash
TODO
```