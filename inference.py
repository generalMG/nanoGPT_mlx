"""
Inference script for generating text with a trained GPT model.

This module provides functionality to load a trained GPT model and generate
text using autoregressive sampling with temperature and top-k filtering.
"""

import argparse
import json
import logging
import os
from typing import Tuple

import mlx.core as mx
import tiktoken

from model import GPT, GPTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for text generation.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate text with a trained GPT-MLX model."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="gpt2_shakespeare_pretrain_mlx",
        help="Directory containing the saved weights/config from train.py.",
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Optional explicit path to the weights .npz file.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional explicit path to the model config .json file.",
    )
    parser.add_argument(
        "--prompt",
        default="To be, or not to be",
        help="Prompt text used to seed generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of tokens to sample after the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k truncation for logits.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Resolve checkpoint paths from arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (weights_path, config_path).

    Raises:
        FileNotFoundError: If weights or config files don't exist.
    """
    ckpt_dir = args.checkpoint_dir
    base = os.path.basename(os.path.normpath(ckpt_dir))
    weights = args.weights_path or os.path.join(ckpt_dir, f"{base}.npz")
    config = args.config_path or os.path.join(ckpt_dir, f"{base}.json")

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Could not find weights file at {weights}. "
            f"Make sure you have trained a model first with train.py"
        )
    if not os.path.exists(config):
        raise FileNotFoundError(
            f"Could not find config file at {config}. "
            f"Make sure you have trained a model first with train.py"
        )
    return weights, config


def load_model(weights_path: str, config_path: str) -> GPT:
    """
    Load a trained GPT model from checkpoint.

    Args:
        weights_path: Path to the .npz weights file.
        config_path: Path to the .json config file.

    Returns:
        Loaded GPT model ready for inference.
    """
    logger.info(f"Loading model config from {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = GPT(GPTConfig(**cfg))

    logger.info(f"Loading model weights from {weights_path}")
    model.load_weights(weights_path, strict=False)
    mx.eval(model.parameters())

    # Log model info
    logger.info(f"Loaded model with config: {cfg}")
    return model


def encode_prompt(prompt_text: str) -> Tuple[tiktoken.Encoding, mx.array]:
    """
    Encode a text prompt into token IDs.

    Args:
        prompt_text: The text prompt to encode.

    Returns:
        Tuple of (tokenizer, token_ids as MLX array).
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(prompt_text)

    if not token_ids:
        logger.warning("Empty prompt, using EOT token")
        token_ids = [tokenizer.eot_token]

    tokens = mx.array([token_ids], dtype=mx.int64)
    logger.info(f"Encoded prompt '{prompt_text}' into {len(token_ids)} tokens")
    return tokenizer, tokens


def generate_text(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    """
    Generate text using the model.

    Args:
        model: The GPT model.
        tokenizer: Tokenizer for decoding.
        prompt_tokens: Input token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Optional top-k filtering.

    Returns:
        Generated text string.
    """
    logger.info(
        f"Generating {max_new_tokens} tokens with temperature={temperature}, top_k={top_k}"
    )

    generated = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_tokens = generated[0].tolist()
    text = tokenizer.decode(generated_tokens)
    return text


def main() -> None:
    """Main entry point for text generation."""
    args = parse_args()

    # Validate temperature
    if args.temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {args.temperature}")

    # Validate top_k
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError(f"top_k must be positive if specified, got {args.top_k}")

    weights_path, config_path = resolve_paths(args)
    model = load_model(weights_path, config_path)
    tokenizer, prompt_tokens = encode_prompt(args.prompt)

    text = generate_text(
        model,
        tokenizer,
        prompt_tokens,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    print("\n" + "=" * 50)
    print("Generated Text:")
    print("=" * 50)
    print(text)
    print("=" * 50)


if __name__ == "__main__":
    main()
