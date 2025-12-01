#!/bin/bash

PROJECT_ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT_DIRECTORY}:${PYTHONPATH}"

python marsfill/cli/dataset.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Falha na execução do pipeline. Código de erro: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Pipeline finalizado com sucesso."
