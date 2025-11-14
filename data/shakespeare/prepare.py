import os
import requests
import tiktoken
import mlx.core as mx
import struct

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = mx.array(train_ids, dtype=mx.uint16)
val_ids = mx.array(val_ids, dtype=mx.uint16)
print(train_ids)

# Function to save array to binary file
def save_array_to_bin(array, file_path):
    with open(file_path, 'wb') as f:
        for val in array.tolist():
            f.write(struct.pack('H', val))

save_array_to_bin(train_ids, os.path.join(os.path.dirname(__file__), 'train.bin'))
save_array_to_bin(val_ids, os.path.join(os.path.dirname(__file__), 'val.bin'))
