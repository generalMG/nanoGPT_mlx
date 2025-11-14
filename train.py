import os
import math
import time
import json
import argparse

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from model import GPTConfig, GPT
from tboard_utils import init_tensorboard, get_tensorboard


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model with MLX")

    # Model architecture
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', type=lambda x: str(x).lower() == 'true', default=False, help='Use bias in linear layers')
    parser.add_argument('--d_type', type=str, default='float32', help='Data type for model parameters')

    # Optimizer settings
    parser.add_argument('--learning_rate', type=float, default=6.0e-4, help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=6.0e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay coefficient')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')

    # Training settings
    parser.add_argument('--num_iters', type=int, default=600000, help='Total number of training iterations')
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='Warmup percentage (deprecated, use warmup_iters)')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='Number of iterations for learning rate decay')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=512, help='Number of gradient accumulation steps')
    parser.add_argument('--context_size', type=int, default=1024, help='Context window size')

    # Data settings
    parser.add_argument('--dataset', type=str, default='shakespeare', help='Dataset name')
    parser.add_argument('--meta_vocab_size', type=int, default=None, help='Vocabulary size (None for default 50304)')

    # Logging and checkpointing
    parser.add_argument('--out_dir', type=str, default='gpt2_shakespeare_pretrain_mlx', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=1, help='Save checkpoint every N iterations')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate every N iterations')
    parser.add_argument('--log_interval', type=int, default=10, help='Log training metrics every N iterations')
    parser.add_argument('--eval_only', type=lambda x: str(x).lower() == 'true', default=False, help='Only run evaluation')

    return parser.parse_args()


# Parse arguments
args = parse_args()

# Extract args to variables for backward compatibility with existing code
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
d_type = args.d_type

learning_rate = args.learning_rate
min_lr = args.min_lr
num_iters = args.num_iters
warmup_pct = args.warmup_pct
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
meta_vocab_size = args.meta_vocab_size

dataset = args.dataset
batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
context_size = args.context_size

save_interval = args.save_interval
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_only = args.eval_only
out_dir = args.out_dir

data_dir = os.path.join('data', dataset)

def load_binary_data(file_path, dtype):
    with open(file_path, 'rb') as f:
        data = f.read()
    return mx.array(memoryview(data).cast(dtype))

train_data = load_binary_data(os.path.join(data_dir, 'train.bin'), 'H')
val_data = load_binary_data(os.path.join(data_dir, 'val.bin'), 'H')

save_model_path = os.path.join(out_dir, out_dir + '.npz')
save_model_config_path = os.path.join(out_dir, out_dir + '.json')

os.makedirs(out_dir, exist_ok=True)
tboard_dir = os.path.join(out_dir, "tboard_log")
init_tensorboard(tboard_dir)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - context_size, shape=(batch_size,)).tolist()
    x = mx.stack([(mx.array(data[i:i+context_size])) for i in ix]).astype(mx.int64)
    y = mx.stack([(mx.array(data[i+1:i+1+context_size])) for i in ix]).astype(mx.int64)
    return x, y

def print_loss(optimizer, iteration_count, average_loss, tic):
    toc = time.perf_counter()
    print(
        f"iter {iteration_count}: train loss {average_loss:.3f}, "
        f"it/sec {1.0 / (toc - tic):.3f}, "
        f"lr {optimizer.learning_rate.item():.9f}"
    )
    return toc

def update_learning_rate(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (
        lr_decay_iters - warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    new_lr = min_lr + coeff * (learning_rate - min_lr)
    return new_lr

def log_tboard_dict(log_dict, itr, pre, post=''):
    writer = get_tensorboard()
    for k, v in log_dict.items():
        writer.add_scalar(f'{pre}/{k}{post}', v, itr)

def main():
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=context_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

    # initialize model:
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model)

    weights = tree_map(lambda p: p.astype(getattr(mx, d_type)), model.parameters())
    model.update(weights)

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")


    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


    # setup optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate,
                            betas=[beta1, beta2],
                            weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


    def step(inputs, targets, gradient_accumulation_steps):
        # gradient accumulation
        accumulated_grads = tree_map(
                    lambda x: mx.zeros_like(x), model.parameters()
                )
        accumulated_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            loss, grads = loss_and_grad_fn(model, inputs, targets)

            accumulated_grads = tree_map(
                lambda acc, new: acc + new * (1.0 / gradient_accumulation_steps),
                accumulated_grads,
                grads,
            )

            tree_map(
                lambda grad: mx.eval(grad),
                accumulated_grads,
            )

            accumulated_loss += loss.item()

        # scale the loss to account for gradient accumulation
        loss = mx.array(accumulated_loss / gradient_accumulation_steps) 

        optimizer.update(model, accumulated_grads)

        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), model.parameters()
        )
        return loss

    # fetch the first batch of samples.
    X, Y = get_batch('train')
    
    state = [model.state, optimizer.state]

    tic = time.perf_counter()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    iter_num = 0
    
    while True:
        if iter_num == 0 and eval_only:
            break

        # lr schedule
        new_lr = update_learning_rate(iter_num)
        optimizer.learning_rate = new_lr

        # mx.simplify(loss, model.parameters())
        loss = step(X, Y, gradient_accumulation_steps)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        tic = print_loss(optimizer, iter_num, loss.item(), tic)

        mx.eval(state)

        if iter_num % log_interval == 0:
            log_train_dict = {
                'loss': loss.item(),
                'lr': new_lr
            }
            log_tboard_dict(log_train_dict, iter_num, 'train')
        
        if iter_num % save_interval == 0:
            # save mode weights
            flat_params = tree_flatten(model.parameters())
            mx.savez(save_model_path, **dict(flat_params))
            # save model config
            with open(save_model_config_path, "w") as f:
                json.dump(model.config.__dict__, f)

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > num_iters:
            break

if __name__ == "__main__":
    main()