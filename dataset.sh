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

# --- 2. Feedback Visual ---
echo "🗺️  Inicializando pipeline de construção do Dataset Marsfill..."

# --- 3. Execução do Script Python ---
# "$@" repassa todos os argumentos (flags) recebidos pelo shell script para o Python
python marsfill/cli/dataset.py "$@"

# --- 4. Captura de Erros ---
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Falha na execução do pipeline. Código de erro: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Pipeline finalizado com sucesso."
