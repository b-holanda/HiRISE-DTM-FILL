#!/bin/bash

# Define o diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Diretório alvo para busca recursiva
TARGET_DIR="${PROJECT_ROOT}/data/dataset/v1/test"

echo "🕳️  Iniciando geração de buracos (Hole Generation) em lote..."
echo "📂 Buscando arquivos .IMG em: $TARGET_DIR"

# Contador para feedback
count=0

# O comando find busca recursivamente arquivos .IMG dentro do diretório alvo
# -print0 e read -d '' garantem que funcione mesmo se houver espaços nos nomes dos arquivos
find "$TARGET_DIR" -type f -name "*.IMG" -print0 | while IFS= read -r -d '' input_file; do
    
    # 1. Determina o nome do arquivo de saída
    # Remove a extensão .IMG e adiciona _with_nodata.tif
    # Ex: arquivo.IMG -> arquivo_with_nodata.tif
    output_file="${input_file%.IMG}_with_nodata.tif"

    # 2. Feedback visual simples
    # Extrai apenas o nome do arquivo para não poluir o log com o caminho completo
    filename=$(basename "$input_file")
    echo "🔨 Processando: $filename"

    # 3. Executa o script Python
    # Passamos -i e -o automaticamente. 
    # "$@" repassa argumentos EXTRAS (ex: tamanho do buraco, seed) que você queira configurar.
    python -m marsfill.cli.hole_gen \
        -i "$input_file" \
        -o "$output_file" \
        "$@"

    # Verifica se houve erro no comando anterior
    if [ $? -ne 0 ]; then
        echo "❌ Erro ao processar $filename"
    fi

    ((count++))
done

if [ $count -eq 0 ]; then
    echo "⚠️  Nenhum arquivo .IMG encontrado em $TARGET_DIR"
else
    echo "✅ Processamento concluído. $count arquivos gerados."
fi
