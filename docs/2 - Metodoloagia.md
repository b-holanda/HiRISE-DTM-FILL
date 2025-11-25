# Preenchimento de Lacunas (Gap-Filling) em DTMs HiRISE usando Inferência Monocular com Vision Transformers (ViT)

## Metodologia

### 1. Aquisição e Preparação do Conjunto de Dados de Treinamento

*Materiais*

- Fonte de Dados: Catálogo STAC do USGS Astrogeology.
- Produtos de Dados:
    - Ortoimagem HiRISE (Asset: ORTHO.tif).
    - DTM HiRISE (Asset: DTM.tif).
- Software: Python, pystac-client (para busca), GDAL/osgeo (para processamento), NumPy
- Hardware: 1 TB

*Metodos*

- Aquisição em Lote: Baixar $N$ (ex: $N=1000$) pares de ORTHO.tif e DTM.tif
- Pré-processamento e Alinhamento:
    - Usar gdalwarp para reamostrar o DTM, alinhando-o perfeitamente à grade da sua Ortoimagem correspondente.
- Extração de Máscara de Lacuna:
    - Para cada DTM alinhado, gerar uma máscara binária identificando todos os pixels com valor NoData.
- Geração de Blocos (Tiling) de Treinamento:
    - Este é o passo crucial. O objetivo é criar blocos de treinamento (ex: 512x512 pixels) que representem **terreno perfeito e sem falhas**.
    - Criar um script que "desliza" uma janela de 512x512 sobre os DTMs alinhados.
    - Condição de Aceitação: Um bloco só é salvo no conjunto de treinamento se ele contiver 100% de pixels válidos (ou seja, 0% de sobreposição com a máscara de lacuna).
    - Se um bloco de DTM é aceito, o bloco correspondente da Ortoimagem é extraído e salvo.
    - O resultado é um grande conjunto de pares (Bloco de Imagem, Bloco de DTM) sem nenhuma lacuna.
- Normalização e Divisão:
    - Imagens (X): Normalizar pixels para $[0, 1]$.
    - DEMs (Y): Normalizar elevação por bloco (Min-Max local) para $[0, 1]$ ou $[-1, 1]$.
    - Dividir os DTMs inteiros em conjuntos de Treinamento (80%), Validação (10%) e Teste (10%), para garantir que o modelo não seja testado em áreas que já viu durante o treino.

### 2. Arquitetura e Treinamento do Modelo

*Materiais*

- Framework: PyTorch.
- Arquitetura: DPT (Density Prediction Transformer).
- Pesos Pré-treinados: Iniciar com pesos do DPT pré-treinado no conjunto MiDaS (dados da Terra), aplicando Transfer Learning.
- Hardware: GPUs de alta VRAM (ex: NVIDIA A100).

*Métodos*

- Arquitetura:
    - Encoder: ViT (ex: ViT-Large) que recebe os blocos de imagem 512x512.
    - Decoder: Módulos de Refinement convolucionais (como no DPT) que reconstroem a informação espacial para gerar o mapa de elevação 512x512.
- Ajuste Fino (Fine-Tuning):
    - Treinar o modelo nos blocos de treinamento da Fase 1.
    - O modelo aprenderá: "Esta textura de sombra e albedo na imagem corresponde a esta forma 3D no DTM."
- Função de Perda (Loss Function) Combinada:
    - A perda deve ser sensível tanto à elevação absoluta quanto à forma do terreno.
    - $L_{\text{total}} = w_1 \cdot L_{\text{L1}} + w_2 \cdot L_{\text{Gradiente/Sobel}} + w_3 \cdot L_{\text{SSIM}}$
    - $L_{\text{L1}}$ (ou L2): Penaliza a diferença de elevação pixel a pixel.
    - $L_{\text{Gradiente}}$: Penaliza a diferença nas inclinações (slopes), forçando o modelo a aprender a inclinação correta das crateras.
    - $L_{\text{SSIM}}$: Garante que a estrutura geral do terreno seja plausível.
- Treinamento: Treinar até que a perda no conjunto de validação pare de melhorar (early stopping).

### 3. Inferência, Pós-processamento e Validação

*Materiais*

- Modelo ViT treinado (.pth).
- Dados do conjunto de Teste:
    - Ortoimagens completas.
    - DTMs completos (com suas lacunas originais).
- Software: GDAL, NumPy, QGIS (para inspeção visual), Scikit-learn (para métricas).

*Métodos*

**1. Fluxo de Inferência**

- Seleção de Alvo: Carregar um par (Ortoimagem, DTM com lacunas) do conjunto de Teste.
- Identificação de Lacunas: Gerar a máscara de NoData para o DTM.
- Extração de Alvos de Imagem:
    - Para cada região de lacuna, extrair o patch de imagem correspondente da Ortoimagem.
    - Contexto é crucial: O patch de imagem extraído deve ser ligeiramente maior que a lacuna (ex: usando o bounding box da lacuna + padding), para que o modelo tenha contexto visual ao redor do buraco.
- Inferência do Modelo: Passar o(s) patch(es) de imagem pelo modelo ViT treinado. O modelo irá gerar "patches de DTM" completos e plausíveis.
- Recomposição e Fusão (Infilling & Blending):
    - Este é o passo de pós-processamento mais importante.
    - Pegar o DTM original (com lacuna).
    - Usar a máscara de NoData para "recortar" os pixels gerados pela IA que correspondem exatamente à área da lacuna.
    - Suavização de Borda (Blending/Feathering): Simplesmente copiar os pixels da IA criará uma "costura" óbvia. É necessário aplicar um algoritmo de fusão (ex: média ponderada em uma zona de transição de 10 pixels, ou um Poisson Blending mais avançado) para suavizar a transição entre os dados reais da fotogrametria e os dados gerados pela IA.
    - O resultado é um DTM "híbrido", onde 99% dos dados são reais e 1% (as lacunas) são preenchidos pela IA.

**2. Validação**

Não podemos validar diretamente os dados que inventamos. Portanto, usamos um teste sintético no conjunto de Teste.

- Criação de Teste Sintético:
    - Pegar um bloco de DTM perfeito (sem lacunas) do conjunto de Teste (que o modelo nunca viu). Este é o nosso "Ground Truth A".
    - Pegar a Imagem correspondente ("Imagem A").
    - Artificialmente, criar um "buraco" (ex: um círculo de NoData) no meio do "Ground Truth A", criando o "DTM com furo B".
- Execução do Pipeline:
    - Alimentar o "DTM com furo B" e a "Imagem A" no pipeline de inferência da Fase 3.
    - O modelo irá detectar o furo artificial e preenchê-lo. O resultado é o "DTM preenchido C".
- Cálculo de Erro:
    - Agora podemos comparar o "Ground Truth A" (o original perfeito) com o "DTM preenchido C" (o reconstruído pela IA).
    - Métricas: Calcular RMSE (em metros), MAE e SSIM apenas dentro da área do buraco artificial.
    - Isso nos dá uma medida quantitativa de quão bem a IA consegue "adivinhar" o terreno real.
- Validação Qualitativa:
    - Inspecionar os DTMs híbridos (os originais, não sintéticos) no QGIS.
    - Gerar hillshades (sombreamentos) e perfis topográficos que cruzam as áreas preenchidas para verificar visualmente se há artefatos, costuras ou transições geologicamente implausíveis.
