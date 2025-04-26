# LongBench Language Modeling

## Prerequisites

## Dependencies

```bash
# Create conda environment
conda create -n thinformer-longbench python=3.12
conda activate thinformer-longbench
pip install torch torchvision torchaudio
# Install triton
pip install triton
# Install other needed Python packages
pip install tqdm einops
```

> \[!NOTE\]
> The original step to install triton using `pip install triton==2.0.0.dev20221202 --no-deps`.

## Results

```bash
python benchmark_single_attention.py --attn_method hyper
```