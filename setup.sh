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
# We use --break-system-packages to bypass PEP 668 on managed environments like newer Ubuntu images
pip install -q --upgrade pip --break-system-packages
pip install -q torch --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
pip install -q datasets transformers bitsandbytes accelerate tqdm --break-system-packages
pip install -q -e .. --break-system-packages

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "✅ Setup complete!"
echo ""
echo "Run training with:"
echo "  torchrun --nproc_per_node=8 -m titans.train --dataset fineweb --d_model 4096"
