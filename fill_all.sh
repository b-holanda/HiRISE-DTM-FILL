#!/bin/bash

# Define o diretório raiz do projeto de forma absoluta
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- Configurações de Caminhos e URLs ---

# 1. Dataset de Teste (Ground Truth)
TEST_DATA_DIR="${PROJECT_ROOT}/data/dataset/v1/test"
GT_URL="https://hirise-dtm-fill.s3.us-east-1.amazonaws.com/ground_truth.zip"
GT_ZIP="ground_truth.zip"

# 2. Modelo Treinado (Pesos .pth)
# Baseado na sua navegação, o diretório de modelos fica dentro de 'data'
MODELS_DIR="${PROJECT_ROOT}/data/models"
MODEL_URL="https://hirise-dtm-fill.s3.us-east-1.amazonaws.com/marsfill_model.pth"
MODEL_FILE="marsfill_model.pth"


# --- BLOCO 1: Preparação do Dataset de Teste ---

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "📂 Diretório de teste não encontrado."
    echo "⚙️  Configurando Ground Truth..."
    
    mkdir -p "$TEST_DATA_DIR"
    cd "$TEST_DATA_DIR" || exit 1

    echo "⬇️  Baixando Ground Truth (Zip)..."
    wget -c --show-progress "$GT_URL" -O "$GT_ZIP"

    echo "📦 Extraindo arquivos..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -o -q "$GT_ZIP" # -q para quiet (menos texto na tela)
    else
        echo "❌ Erro: 'unzip' não instalado."
        exit 1
    fi
    
    # Opcional: rm "$GT_ZIP"
    echo "✅ Dataset de teste pronto."
else
    echo "📂 Dataset de teste já existe. Verificado."
fi

echo "---------------------------------------------------"


# --- BLOCO 2: Download do Modelo Treinado ---

# Verifica se o arquivo do modelo JÁ existe para não baixar de novo à toa
if [ ! -f "${MODELS_DIR}/${MODEL_FILE}" ]; then
    echo "🤖 Modelo treinado não encontrado."
    echo "⚙️  Baixando pesos do modelo..."

    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR" || exit 1

    echo "⬇️  Baixando ${MODEL_FILE}..."
    wget -c --show-progress "$MODEL_URL" -O "$MODEL_FILE"

    echo "✅ Download do modelo concluído."
else
    echo "🤖 Modelo encontrado em: data/models/${MODEL_FILE}"
fi

echo "---------------------------------------------------"


# --- BLOCO 3: Execução da Inferência ---

# Garante que estamos na raiz do projeto antes de rodar o Python
cd "$PROJECT_ROOT" || exit 1

chmod a+x hole_gen.sh

./hole_gen.sh

echo "🔮 Inicializando pipeline de inferência (Fill)..."

python marsfill/cli/batch_validate.py

# --- Captura de Erros ---
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Falha na inferência. Código de erro: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Inferência finalizada com sucesso."
