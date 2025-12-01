#!/bin/bash

# Define o diretório raiz do projeto
PROJECT_ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT_DIRECTORY}:${PYTHONPATH}"

# Configurações do Dataset
DATA_DIR="${PROJECT_ROOT_DIRECTORY}/data"
DATASET_URL="https://hirise-dtm-fill.s3.us-east-1.amazonaws.com/dataset.tar"
TAR_FILE="dataset.tar"

# --- 1. Verificação e Preparação do Ambiente (Automated Setup) ---

if [ ! -d "$DATA_DIR" ]; then
    echo "📂 Diretório 'data' não encontrado. Iniciando configuração automática..."
    
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR" || exit 1

    echo "⬇️  Baixando dataset (860.0 GB)... Isso pode demorar."
    # wget com -c (continue) para retomar downloads falhos e --show-progress para barra visual
    wget -c --show-progress "$DATASET_URL" -O "$TAR_FILE"

    echo "📦 Extraindo arquivos..."
    
    # Lógica para Barra de Progresso na Descompressão
    if command -v pv >/dev/null 2>&1; then
        # Se 'pv' estiver instalado, usa para mostrar barra de progresso baseada no tamanho
        pv "$TAR_FILE" | tar -xf -
    else
        # Fallback se não tiver 'pv': usa tar verbose padrão
        echo "⚠️  'pv' não encontrado para barra de progresso. Instalando 'sudo apt install pv' ficaria mais bonito."
        echo "   Usando modo verbose padrão..."
        tar -xvf "$TAR_FILE"
    fi

    # Opcional: Remover o tar após extrair para economizar espaço
    # rm "$TAR_FILE"
    
    cd "$PROJECT_ROOT_DIRECTORY" || exit 1
    echo "✅ Setup de dados concluído."
else
    echo "📂 Diretório 'data' já existe. Pulando download."
fi

echo "---------------------------------------------------"


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
