#!/bin/bash

# Define o diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &amp;&amp; pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

# Exporta variáveis de ambiente
export SM_CHANNEL_TRAIN=/dataset/train
export SM_CHANNEL_VALIDATION=/dataset/validation
export SM_MODEL_DIR=/dataset/model

if [ $NUM_GPUS -gt 1 ]; then
    echo "🚀 Treinando com $NUM_GPUS GPUs (DDP)"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        marsfill/cli/train.py
else
    echo "🖥️  Treinando com 1 GPU"
    python marsfill/cli/train.py
fi
