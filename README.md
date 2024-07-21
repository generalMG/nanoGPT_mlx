# GPT-MLX Training and Data Preparation

## Overview

This repository contains scripts for training a GPT model using the MLX library and preparing the dataset.

## Setup

1. Clone the repository:

```
git clone https://github.com/generalMG/nanoGPT_mlx.git
```

## Data Preparation

The prepare.py script downloads and processes the Tiny Shakespeare dataset.

1. Navigate to the data directory:

```
cd data/shakespeare
```

2. Run the script:

```
python prepare.py
```

## Training

The train.py script trains the GPT model using the MLX library.

1. Run the training script:

```
python train.py
```

Note: The code was taken from https://github.com/vithursant/nanoGPT_mlx and some edits were made in order to convert the code to full mlx support.
The code was written for learning purposes.
