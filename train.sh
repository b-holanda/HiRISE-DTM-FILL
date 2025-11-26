#!/bin/bash

# =================================================================================================
# SCRIPT DE ORQUESTRAÇÃO DE TREINAMENTO MARSFILL
#
# Descrição:
#   Configura o ambiente de execução, define o PYTHONPATH e detecta o hardware disponível.
#   Decide automaticamente entre execução distribuída (torchrun) ou execução simples (python)
#   baseado no número de GPUs detectadas.
#
# Argumentos:
#   $@ : Todos os argumentos passados para este script serão encaminhados para o train.py
#        (ex: --profile dev, --profile prod)
#
# Variáveis de Ambiente (Opcionais - Defaults definidos abaixo):
#   SM_CHANNEL_TRAIN      : Caminho dos dados de treino.
#   SM_CHANNEL_VALIDATION : Caminho dos dados de validação.
#   SM_MODEL_DIR          : Diretório onde o modelo final será salvo.
# =================================================================================================

PROJECT_ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT_DIRECTORY}:${PYTHONPATH}"

# Define defaults apenas se as variaveis nao estiverem definidas no ambiente
export SM_CHANNEL_TRAIN="${SM_CHANNEL_TRAIN:-data/dataset/v1/train}"
export SM_CHANNEL_VALIDATION="${SM_CHANNEL_VALIDATION:-data/dataset/v1/validation}"
export SM_MODEL_DIR="${SM_MODEL_DIR:-data/models}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

AVAILABLE_GPU_COUNT=0

if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
fi

if [ "$AVAILABLE_GPU_COUNT" -gt 1 ]; then
    echo "🚀 Modo Detectado: Treinamento Distribuído (DDP) com $AVAILABLE_GPU_COUNT GPUs"
    
    torchrun \
        --nproc_per_node="$AVAILABLE_GPU_COUNT" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        marsfill/cli/train.py "$@"

else
    if [ "$AVAILABLE_GPU_COUNT" -eq 1 ]; then
        echo "🖥️  Modo Detectado: GPU Única"
    else
        echo "🐌 Modo Detectado: CPU (Atenção: Lento para treinamento)"
    fi

    python marsfill/cli/train.py "$@"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ O treinamento falhou com código de erro: $EXIT_CODE"
    exit $EXIT_CODE
fi
