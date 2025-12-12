#!/bin/bash
# Titans Quick Setup for Cloud GPUs (RunPod, Lium, Lambda, etc.)
# Usage: curl -fsSL https://raw.githubusercontent.com/YOUR_USER/agitrust/main/setup.sh | bash

set -e

echo "🚀 Setting up Titans..."

# Clone or update repo
if [ -d "titans" ]; then
    cd titans && git pull
else
    git clone https://github.com/tlgkxd123/titans.git
    cd titans
fi

# Install dependencies (no venv needed on cloud instances)
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
pip install -q datasets transformers bitsandbytes accelerate tqdm

# Install titans package
pip install -q -e .

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "✅ Setup complete!"
echo ""
echo "Run training with:"
echo "  torchrun --nproc_per_node=8 -m titans.train --dataset fineweb --d_model 4096"
