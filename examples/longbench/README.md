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
pip install tqdm
```

> \[!NOTE\]
> The original instructions for installing triton using `pip install triton==2.0.0.dev20221202 --no-deps` from the `hyper-attn` repo is outdated. If installing `torch==2.6.0`, triton 3.2.0 will automatically be installed.

## Results

```bash
python benchmark_single_attention.py --attn_method flash-cuda
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
