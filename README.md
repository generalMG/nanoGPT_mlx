# GPT-MLX Training, Data Preparation, and Inference

## Overview

This project trains a GPT-style model on Tiny Shakespeare using MLX. It now ships with a requirements file, CLI overrides for training hyperparameters, and a dedicated inference script for sampling from saved checkpoints.

## Environment & Dependencies

1. (Optional) Create or activate a Python environment (e.g. `source ~/workspace/envs/selfdev_macos/bin/activate`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

The Tiny Shakespeare dataset and tokenizer assets are prepared with `prepare.py`.

```
python data/shakespeare/prepare.py
```

This writes `train.bin` / `val.bin` under `data/shakespeare/`.

## Training

`train.py` consumes the prepared binaries and saves checkpoints under `out_dir` (default `gpt2_shakespeare_pretrain_mlx`). Override any config value on the CLI by passing `--key=value`; for example, a short debug run:

```
python train.py \
  --num_iters=10 \
  --gradient_accumulation_steps=1 \
  --context_size=64 \
  --save_interval=10 \
  --log_interval=1
```

Artifacts produced inside `out_dir`:
- `<out_dir>.npz`: model weights saved via `mx.savez`.
- `<out_dir>.json`: serialized `GPTConfig`.
- `tboard_log/`: TensorBoard summaries if enabled.

## Inference

Use the `inference.py` entry point to load checkpoints and generate text.

```
python inference.py \
  --checkpoint_dir gpt2_shakespeare_pretrain_mlx \
  --prompt "O Romeo" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50
```

`--weights_path`/`--config_path` can be provided directly if the files were moved, and prompts are encoded/decoded with `tiktoken`'s GPT‑2 vocabulary.

## Notes

- The code is adapted from https://github.com/vithursant/nanoGPT_mlx with modifications for full MLX support and new tooling around training/inference.
- Scripts assume macOS with MLX-compatible hardware (Apple Silicon) and Metal support.
