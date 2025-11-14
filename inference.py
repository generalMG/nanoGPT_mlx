import argparse
import json
import os

import mlx.core as mx
import tiktoken

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT-MLX model.")
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
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k truncation for logits.",
    )
    return parser.parse_args()


def resolve_paths(args):
    ckpt_dir = args.checkpoint_dir
    base = os.path.basename(os.path.normpath(ckpt_dir))
    weights = args.weights_path or os.path.join(ckpt_dir, f"{base}.npz")
    config = args.config_path or os.path.join(ckpt_dir, f"{base}.json")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Could not find weights file at {weights}")
    if not os.path.exists(config):
        raise FileNotFoundError(f"Could not find config file at {config}")
    return weights, config


def load_model(weights_path, config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = GPT(GPTConfig(**cfg))
    model.load_weights(weights_path, strict=False)
    mx.eval(model.parameters())
    return model


def encode_prompt(prompt_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(prompt_text)
    if not token_ids:
        token_ids = [tokenizer.eot_token]
    tokens = mx.array([token_ids], dtype=mx.int64)
    return tokenizer, tokens


def main():
    args = parse_args()
    weights_path, config_path = resolve_paths(args)
    model = load_model(weights_path, config_path)
    tokenizer, prompt_tokens = encode_prompt(args.prompt)

    generated = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_tokens = generated[0].tolist()
    text = tokenizer.decode(generated_tokens)
    print(text)


if __name__ == "__main__":
    main()
