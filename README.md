# nanoGPT-MLX

A minimal, educational GPT implementation for Apple Silicon using MLX. Train GPT-style language models efficiently on your Mac with M-series chips.

## Overview

This project implements a GPT-style transformer model from scratch using Apple's MLX framework. It's designed for training on modest datasets (like Tiny Shakespeare) and includes:

- **Pure MLX implementation** - Uses MLX's built-in `nn.LayerNorm`, `mlx.optimizers.AdamW`, and other optimized components
- **Configurable architecture** - All hyperparameters controllable via command-line arguments with sensible defaults
- **Training pipeline** - Complete training loop with gradient accumulation, learning rate scheduling, and TensorBoard logging
- **Inference script** - Generate text from trained checkpoints with temperature and top-k sampling
- **Apple Silicon optimized** - Leverages unified memory and Metal acceleration

## Requirements

- macOS with Apple Silicon (M-series)
- Python 3.8+
- MLX 0.29.4 or later

## Quick Start

### 1. Installation

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Prepare the Tiny Shakespeare dataset:
```bash
python data/shakespeare/prepare.py
```

This downloads the dataset and creates `train.bin` and `val.bin` files in `data/shakespeare/`.

### 3. Train a Model

**Quick debug run** (2-layer, small model):
```bash
python train.py \
  --n_layer=2 \
  --n_head=4 \
  --n_embd=128 \
  --num_iters=1000 \
  --gradient_accumulation_steps=1 \
  --context_size=256
```

**Full training run** (default GPT-2 small config):
```bash
python train.py
```

This will train a 12-layer, 768-dimensional model with ~155M parameters. Training progress is logged to console and TensorBoard.

### 4. Generate Text

After training, generate text from your checkpoint:
```bash
python inference.py \
  --checkpoint_dir gpt2_shakespeare_pretrain_mlx \
  --prompt "To be, or not to be" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 50
```

## Training Configuration

All hyperparameters have defaults and can be overridden via command-line arguments.

### View All Options
```bash
python train.py --help
```

### Key Arguments

#### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--n_layer` | 12 | Number of transformer layers |
| `--n_head` | 12 | Number of attention heads |
| `--n_embd` | 768 | Embedding dimension |
| `--dropout` | 0.0 | Dropout rate |
| `--bias` | False | Use bias in linear layers |
| `--context_size` | 1024 | Context window size |

#### Optimizer Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate` | 6e-4 | Peak learning rate |
| `--min_lr` | 6e-5 | Minimum learning rate |
| `--weight_decay` | 0.1 | Weight decay coefficient |
| `--beta1` | 0.9 | Adam beta1 |
| `--beta2` | 0.95 | Adam beta2 |

#### Training Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_iters` | 600000 | Total training iterations |
| `--batch_size` | 1 | Batch size |
| `--gradient_accumulation_steps` | 512 | Gradient accumulation steps |
| `--warmup_iters` | 2000 | Warmup iterations |
| `--lr_decay_iters` | 600000 | LR decay iterations |

#### Logging & Checkpointing
| Argument | Default | Description |
|----------|---------|-------------|
| `--out_dir` | gpt2_shakespeare_pretrain_mlx | Output directory |
| `--save_interval` | 1 | Save checkpoint every N iterations |
| `--log_interval` | 10 | Log metrics every N iterations |
| `--eval_interval` | 10 | Evaluate every N iterations |

### Training Examples

**Small model for experimentation:**
```bash
python train.py \
  --n_layer=6 \
  --n_head=6 \
  --n_embd=384 \
  --num_iters=10000 \
  --context_size=512
```

**Large model (GPT-2 medium-ish):**
```bash
python train.py \
  --n_layer=24 \
  --n_head=16 \
  --n_embd=1024 \
  --gradient_accumulation_steps=1024
```

**Custom learning rate schedule:**
```bash
python train.py \
  --learning_rate=3e-4 \
  --min_lr=3e-5 \
  --warmup_iters=1000 \
  --lr_decay_iters=50000
```

## Inference Options

The `inference.py` script supports various generation parameters:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | gpt2_shakespeare_pretrain_mlx | Directory with model checkpoint |
| `--weights_path` | None | Direct path to weights file (.npz) |
| `--config_path` | None | Direct path to config file (.json) |
| `--prompt` | "To be, or not to be" | Text prompt for generation |
| `--max_new_tokens` | 100 | Number of tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (higher = more random) |
| `--top_k` | None | Top-k sampling (None = disabled) |

### Inference Examples

**Conservative generation (low temperature):**
```bash
python inference.py \
  --prompt "ROMEO:" \
  --max_new_tokens 150 \
  --temperature 0.5
```

**Creative generation (high temperature):**
```bash
python inference.py \
  --prompt "Once upon a time" \
  --max_new_tokens 300 \
  --temperature 1.2 \
  --top_k 100
```

## Project Structure

```
nanoGPT_mlx/
├── model.py              # GPT model architecture
├── train.py              # Training script with argparse config
├── inference.py          # Text generation script
├── tboard_utils.py       # TensorBoard utilities
├── requirements.txt      # Python dependencies
├── data/
│   └── shakespeare/
│       └── prepare.py    # Dataset preparation script
└── gpt2_shakespeare_pretrain_mlx/  # Default output directory
    ├── *.npz            # Model weights
    ├── *.json           # Model config
    └── tboard_log/      # TensorBoard logs
```

## Model Architecture

The implementation includes:

- **Causal self-attention** with multi-head attention
- **Layer normalization** using MLX's optimized `nn.LayerNorm`
- **MLP blocks** with GELU activation
- **Rotary positional embeddings** via learned position embeddings
- **Gradient accumulation** for effective large batch training
- **Cosine learning rate decay** with warmup

## Recent Updates

- ✅ **MLX 0.29.4 compatibility** - Updated to use latest MLX features
- ✅ **Built-in components** - Replaced custom implementations with MLX's optimized `nn.LayerNorm` and `mlx.optimizers.AdamW`
- ✅ **Argparse configuration** - Clean command-line argument handling with defaults
- ✅ **No NumPy dependency** - Pure MLX implementation
- ✅ **Modern patterns** - Uses `nn.value_and_grad()` and direct `optimizer.learning_rate` assignment
- ✅ **Reproducibility** - Fixed random seed for consistent training results

## Monitoring Training

View training metrics in TensorBoard:
```bash
tensorboard --logdir gpt2_shakespeare_pretrain_mlx/tboard_log
```

Then open http://localhost:6006 in your browser.

## Performance Tips

1. **Gradient Accumulation**: Use `--gradient_accumulation_steps` to simulate larger batches without running out of memory
2. **Context Size**: Reduce `--context_size` for faster iteration during development
3. **Model Size**: Scale `--n_layer`, `--n_head`, and `--n_embd` together for balanced models
4. **Metal Optimization**: MLX automatically optimizes for Metal - no manual configuration needed

## Known Limitations

- Training is currently single-GPU (one Apple Silicon chip)
- Batch size is typically 1 with gradient accumulation used for effective larger batches
- Designed for educational purposes and modest datasets

## Troubleshooting

**Out of memory errors:**
- Reduce `--context_size`
- Reduce `--n_layer`, `--n_head`, or `--n_embd`
- Ensure `--gradient_accumulation_steps=1` for debugging

**Slow training:**
- Check that you're running on Apple Silicon (not Intel Mac)
- Ensure MLX is properly installed: `python -c "import mlx.core as mx; print(mx.__version__)"`
- Monitor Activity Monitor for Metal GPU usage

**Compatibility Note:**
- Tested and works great on Apple Silicon (M-series) machines

## Citation

This code is adapted from [vithursant/nanoGPT_mlx](https://github.com/vithursant/nanoGPT_mlx) with significant modifications:
- Full MLX 0.29.4 compatibility
- Built-in MLX component usage (LayerNorm, AdamW)
- Argparse-based configuration system
- Enhanced documentation and examples

Original nanoGPT by Andrej Karpathy: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## License

MIT License - See [LICENSE](LICENSE) file for details.
