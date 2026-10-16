#!/bin/bash
# ==============================================================================
#            Setup Conda Environment for Enhanced RRSIS on Linux (RTX 4090 / 4080)
# ==============================================================================

set -e  # Exit immediately on error

echo "=============================================================================="
echo "🚀 Step 1: Checking NVIDIA GPU & Driver..."
echo "=============================================================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️  [WARNING] nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
fi

ENV_NAME="sam3"

echo ""
echo "=============================================================================="
echo "📦 Step 2: Creating Conda Environment '${ENV_NAME}' with Python 3.10..."
echo "=============================================================================="
# Source conda into bash subshell
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}

echo ""
echo "=============================================================================="
echo "🔥 Step 3: Installing PyTorch with CUDA 12.4 (Optimized for RTX 4090 / 4080)..."
echo "=============================================================================="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=============================================================================="
echo "📚 Step 4: Installing Project Dependencies..."
echo "=============================================================================="
pip install -r requirements.txt
pip install transformers omegaconf yacs addict ftfy pyyaml

echo ""
echo "=============================================================================="
echo "🔍 Step 5: Verifying Installation & GPU Detection..."
echo "=============================================================================="
python -c "
import torch
print('='*50)
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device Name  : {torch.cuda.get_device_name(0)}')
    print(f'Device Count     : {torch.cuda.device_count()}')
    print(f'CUDA Capability  : {torch.cuda.get_device_capability(0)}')
print('='*50)
"

echo "🎉 Environment '${ENV_NAME}' successfully setup and ready!"
echo "👉 To activate it anytime, run: conda activate ${ENV_NAME}"
