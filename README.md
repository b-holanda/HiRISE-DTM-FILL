# marsfill

-----

## Visão Geral

`marsfill` é um pipeline que pega pares de produtos HiRISE (uma ortoimagem e seu DTM com buracos) e treina um modelo de IA para prever o relevo onde a fotogrametria falhou. Ele funciona em dois modos:
- **local**: lê/escreve tudo na pasta `./data`.
- **s3**: lê/escreve direto no bucket (`s3://hirise-dtm-fill` por padrão).

O fluxo completo é:
1) Buscar pares DTM+ORTHO públicos do PDS HiRISE, alinhar e cortar em blocos 512×512 sem lacunas, salvando em Parquet (treino/validação) e também guardando os pares de teste integrais.
2) Treinar um modelo baseado no `Intel/dpt-large` (Vision Transformer para profundidade) com perdas L1 + gradiente + SSIM.
3) Rodar inferência por blocos nas áreas NoData, suavizar bordas e gerar métricas/plots. As saídas são salvas no mesmo modo (local ou S3).

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

Todo o pipeline funciona em dois modos: **s3** (dados lidos/escritos direto no bucket) ou **local** (dados em `./data` na raiz do projeto). Os comandos abaixo assumem o perfil `prod`; use `--profile test` para o perfil de teste.

Diagramas (PlantUML) estão em `docs/diagrams/`:
- `dataset_sequence.puml`
- `train_sequence.puml`
- `fill_sequence.puml`

### Testes

Após criar/atualizar o ambiente Conda, rode:

```bash
conda activate marsfill-env
pytest
```

### 1) Gerar rótulos de treinamento (100 pares DTM+ORTHO)

Requisitos recomendados:
- **Memória RAM**: 32 GB
- **Núcleos de CPU**: 4
- **Espaço em disco**: 2 TB

```bash
# S3
./dataset.sh --profile prod --mode s3
# Local
./dataset.sh --profile prod --mode local
```

Saídas esperadas:
- Treino: `s3://hirise-dtm-fill/dataset/v1/train` ou `./data/dataset/v1/train`
- Validação: `s3://hirise-dtm-fill/dataset/v1/validation` ou `./data/dataset/v1/validation`
- Teste (arquivos integrais): `s3://hirise-dtm-fill/dataset/v1/test/test-a/{dtm.IMG, ortho.JP2}`, `test-b`, ... ou `./data/dataset/v1/test/...`

### 2) Treinar o modelo

Requisitos recomendados (Intel/DPT-ViT-Large):
- **GPU**: 4× NVIDIA A10G 24 GB
- **Memória de GPU**: 96 GB
- **Memória RAM**: 192 GB
- **Núcleos de CPU**: 48
- **Espaço em disco**: 2 TB

```bash
# S3
./train.sh --profile prod --mode s3
# Local
./train.sh --profile prod --mode local
```

Entradas: `dataset/v1/train` e `dataset/v1/validation` no bucket ou em `./data`.  
Saída: `s3://hirise-dtm-fill/models/marsfill_model.pth` ou `./data/models/marsfill_model.pth`.

### 3) Executar o modelo (preencher lacunas)

Requisitos recomendados:
- **GPU**: 1× NVIDIA T4
- **Memória de GPU**: 24 GB
- **Memória RAM**: 16 GB
- **Núcleos de CPU**: 4
- **Espaço em disco**: 100 GB

```bash
# S3 (usa par test-a, test-b, ...)
./fill.sh --test a --profile prod --mode s3

# Local
./fill.sh --test a --profile prod --mode local
```

Entradas:
- Modelo: `s3://hirise-dtm-fill/models/marsfill_model.pth` ou `./data/models/marsfill_model.pth`
- Dados de teste: `s3://hirise-dtm-fill/dataset/v1/test/test-a/{dtm.IMG,ortho.JP2}` ou `./data/dataset/v1/test/test-a/...`

Saídas:
- DTM preenchido: `s3://hirise-dtm-fill/filled/test-a/predicted_dtm.tif` ou `./data/filled/test-a/predicted_dtm.tif`
- Máscara: `.../mask_predicted_dtm.tif`
- Métricas: `.../metrics.json`
- Gráficos: `.../result_*.jpg`

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
