#!/bin/bash

# =================================================================================================
# SCRIPT DE CONSTRUÇÃO DO DATASET (ETL)
#
# Descrição:
#   Configura o PYTHONPATH e executa o pipeline de download, processamento e empacotamento
#   das imagens HiRISE.
#
# Uso:
#   ./make_dataset.sh --profile <nome_perfil> --mode <local|s3>
#
# Exemplos:
#   ./make_dataset.sh --profile dev --mode local   (Teste rápido local)
#   ./make_dataset.sh --profile prod --mode s3     (Produção com upload para nuvem)
# =================================================================================================

# 1. Configuração do Diretório Raiz
# Garante que o Python encontre o módulo 'marsfill' independente de onde o script é chamado
PROJECT_ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT_DIRECTORY}:${PYTHONPATH}"

# 2. Feedback Visual
echo "🗺️  Inicializando pipeline de construção do Dataset Marsfill..."

# 3. Execução do Script Python
# "$@" repassa todos os argumentos (flags) recebidos pelo shell script para o Python
python marsfill/cli/build.py "$@"

# 4. Captura de Erros
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Falha na geração do dataset. Código de erro: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Pipeline de dados finalizado com sucesso."
