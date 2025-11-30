# marsfill ![CI](https://img.shields.io/github/actions/workflow/status/b-holanda/HiRISE-DTM-FILL/coverage.yml?branch=main&label=tests%2Fcoverage) ![Coverage Pages](https://img.shields.io/badge/GitHub%20Pages-coverage-blue)

[Página do projeto (coverage report)](https://b-holanda.github.io/HiRISE-DTM-FILL/)

-----

## Visão Geral

`marsfill` é um pipeline que pega pares de produtos HiRISE (uma ortoimagem e seu DTM com buracos) e treina um modelo de IA para prever o relevo onde a fotogrametria falhou. O ETL, o treinamento e agora o preenchimento operam apenas em modo **local** (pasta `./data`).

O fluxo completo é:
1) Buscar pares DTM+ORTHO públicos do PDS HiRISE, alinhar e cortar em blocos 512×512 sem lacunas, salvando em Parquet (treino/validação) e também guardando os pares de teste integrais.
2) Treinar um modelo baseado no `Intel/dpt-large` (Vision Transformer para profundidade) com perdas L1 + gradiente + SSIM.
3) Rodar inferência por blocos nas áreas NoData, suavizar bordas e gerar métricas/plots. As saídas são salvas no mesmo modo (local ou S3).

## Arquitetura

- **Visão Geral**  
  A arquitetura do `marsfill` é organizada em três camadas principais:
  1. **Pipeline de Dados (ETL)**: localiza pares HiRISE (DTM + ORTHO) no repositório público, baixa, alinha, recorta em blocos e normaliza, salvando tudo em um formato eficiente (Parquet) para treino/validação e guardando também alguns pares integrais de teste.
  2. **Treinamento de Modelo (DPT-ViT)**: usa os blocos gerados para fazer fine-tuning de um modelo de profundidade monocular baseado em Vision Transformer (`Intel/dpt-large`), com uma função de perda que combina erro por pixel, gradiente de relevo e similaridade estrutural.
  3. **Preenchimento de DTMs (Inference + Pós-processamento)**: aplica o modelo treinado sobre DTMs com lacunas, trabalhando em tiles com padding de contexto, recalibra as predições para o intervalo real do DTM original e faz um blending suave nas bordas para evitar costuras visíveis.

  Todo esse fluxo é parametrizado por perfis YAML (ex.: `prod`, `test`) e opera localmente na pasta `./data`.

- **Sequência do Pipeline**  
  ![Dataset Sequence](docs/images/dataset_sequence.png)  
  *ETL local: varredura do PDS HiRISE, download/alinhamento com GDAL, tiling sem NoData e salvamento em Parquet (train/val).*
  ![Train Sequence](docs/images/train_sequence.png)  
  *Treinamento local: carga de perfil, criação de DataLoaders streaming, loop de épocas com perda combinada e checkpoint em `data/models`.*
  ![Fill Sequence](docs/images/fill_sequence.png)
  *Inferência de preenchimento: leitura DTM/Ortho local, inferência por blocos com padding, blending das bordas, geração de máscara, métricas e plots; outputs gravados em `./data/filled`.*

- **Backbone de Profundidade (DPT-ViT)**  
  ![DPT Architecture](docs/images/dpt_architecture.jpg)
  *Diagrama conceitual do DPT baseado em ViT: encoder Transformer para contexto global e decoder de refinamento para produzir mapa de profundidade denso, usado como backbone do marsfill.*

## Resultados

- Resumo detalhado e figuras em `docs/results/README.md`.
- Destaques:
  - Construção do dataset: distribuição e tempo de geração de tiles válidos (`docs/images/build_dataset_*.png`).
  - Alinhamento e tiling: exemplos de DTM bruto vs. alinhado e blocos 512×512 (`docs/images/dtm_*`, `docs/images/tile_*`).
  - Testes: 21/21 passando; cobertura ~67% com relatório em GitHub Pages.


## Instalação

```bash
git clone https://github.com/b-holanda/HiRISE-DTM-FILL.git

cd HiRISE-DTM-FILL

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sudo chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

source  ~/miniconda3/bin/activate

conda env create -n marsfill-env -f environment.yml
# Se já existir, atualize:
# conda env update -n marsfill-env -f environment.yml --prune

conda activate marsfill-env

```

## Uso

A geração do dataset, o treinamento e o preenchimento rodam apenas em modo local (pasta padrão `./data`). Os comandos abaixo assumem o perfil `prod`; use `--profile test` para o perfil de teste.

### Testes

Após criar/atualizar o ambiente Conda, rode:

```bash
conda activate marsfill-env
pytest

# Com report de coverage (terminal + HTML em htmlcov/)
pytest --cov --cov-report=term-missing --cov-report=html
```

### Lint/Format (Ruff)

```bash
conda activate marsfill-env
# Formatar código
ruff format .
# Checar lint
ruff check .
```

### 1) Gerar rótulos de treinamento (100 pares DTM+ORTHO)

Requisitos recomendados:
- **Memória RAM**: 32 GB
- **Núcleos de CPU**: 4
- **Espaço em disco**: 2 TB

```bash
./dataset.sh --profile prod
```

Saídas esperadas (local):
- Treino: `./data/dataset/v1/train`
- Validação: `./data/dataset/v1/validation`
- Teste (arquivos integrais): `./data/dataset/v1/test/test-a/{dtm.IMG, ortho.JP2}`, `test-b`, ...
- Assets compactados: `./data/dataset/v1/assets/{train,validation,test}.zip`

> Caso precise usar os dados em um bucket, faça o upload manual do diretório gerado ou dos arquivos `.zip` em `assets/`.

### 2) Treinar o modelo

Requisitos recomendados (Intel/DPT-ViT-Large):
- **GPU**: 4× NVIDIA A10G 24 GB
- **Memória de GPU**: 96 GB
- **Memória RAM**: 192 GB
- **Núcleos de CPU**: 48
- **Espaço em disco**: 2 TB

```bash
./train.sh --profile prod
```

Entradas: `./data/dataset/v1/train` e `./data/dataset/v1/validation`.  
Saída: `./data/models/marsfill_model.pth`.

### 3) Executar o modelo (preencher lacunas)

Requisitos recomendados:
- **GPU**: 1× NVIDIA T4
- **Memória de GPU**: 24 GB
- **Memória RAM**: 16 GB
- **Núcleos de CPU**: 4
- **Espaço em disco**: 100 GB

```bash
./fill.sh --profile prod \
  --dtm data/dataset/v1/test/dunes/DTEPC_088676_2540_088162_2540_A01_with_nodata.IMG \
  --ortho data/dataset/v1/test/dunes/ESP_088676_2540_RED_A_01_ORTHO.JP2 \
  --out_dir data/filled/dunes
```

Entradas:
- Modelo: `data/models/marsfill_model.pth`
- DTM: `data/dataset/v1/test/dunes/DTEPC_088676_2540_088162_2540_A01_with_nodata.IMG`
- ORTHOIMAGE: `data/dataset/v1/test/dunes/ESP_088676_2540_RED_A_01_ORTHO.JP2`

Saídas:
- DTM preenchido: `data/filled/dunes/DTEPC_088676_2540_088162_2540_A01_filled.tif`
- Máscara: `data/filled/dunes/DTEPC_088676_2540_088162_2540_A01_filled_mask.tif`
- Métricas: `data/filled/dunes/metrics.json`
- Gráficos: `data/filled/dunes/result_*.jpg`

### Utilitário: Gerar Buracos (NoData) sintéticos

Crie variações de DTMs com lacunas para testes/controlados:

```bash
./hole_gen.sh \
  -i data/dataset/v1/test/dunes/DTEPC_088676_2540_088162_2540_A01.IMG \
  -o data/dataset/v1/test/dunes/DTEPC_088676_2540_088162_2540_A01_with_nodata.tif
```

O script insere buracos circulares de NoData aleatórios e salva um novo GeoTIFF.

-----

## Licença

Este projeto é licenciado sob a **Licença MIT**. Veja o arquivo `LICENSE` para mais detalhes.

## Como Citar

Se você usar `marsfill` em sua pesquisa, por favor, cite este trabalho:

```bibtex
@misc{marsfill_2025,
  author = {Bruno Rodrigues Holanda},
  title = {marsfill: Reconstrução de DTMs HiRISE com Vision Transformers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/b-holanda/HiRISE-DTM-FILL}}
}
```
