# Resultados Experimentais — marsfill

## Resumo
Este documento sumariza os principais resultados obtidos a partir do pipeline descrito na Metodologia: construção do dataset, qualidade do alinhamento, amostras de tiling e estado atual de testes/coverage. As figuras abaixo foram extraídas dos artefatos gerados no projeto (`docs/images`).

## Métodos em Breve
- **Dataset:** varredura do PDS HiRISE, alinhamento via GDAL, corte em tiles 512×512 sem NoData, exportação em Parquet (train/val/test) e ZIP de assets.
- **Modelo:** fine-tuning do backbone `Intel/dpt-large` com perdas L1 + Gradiente + SSIM.
- **Fill:** inferência por blocos com padding de contexto, desnormalização pelo DTM original e blending de bordas; métricas e plots salvos junto às saídas.

## Resultados

### Construção do Dataset
- **build_dataset_result.png:** distribuição de tiles válidos gerados a partir dos pares DTM/ORTHO; evidencia a filtragem de blocos com 0% NoData.
- **build_dataset_time.png:** tempo de processamento por etapa (download, warp, tiling) no perfil atual.

### Alinhamento e Pré-processamento
- **dtm_puro.png:** DTM original com lacunas (NoData).
- **dtm_alinhado.png:** DTM alinhado à ortoimagem após `gdal.Warp` (grade comum e resolução match).
- **imagem_orthoretificada.png:** ortoimagem já retificada, usada como entrada para inferência.

### Tiling e Amostras
- **tile_dtm.png:** bloco de DTM totalmente válido (usado em treino/val).
- **tile_ortho.png:** bloco da orto correspondente ao tile de DTM.
- **dtm_title_stride.png:** ilustração de passo (stride) entre tiles na extração 512×512.

### Avaliação e Qualidade
- O pipeline de testes automatizados (21/21) está verde; cobertura global ~67% com relatório em GitHub Pages.
- Para novos preenchimentos, métricas (RMSE/MAE/SSIM) e gráficos são gerados por execução em `filled/<test-id>/metrics.json` e `result_comparison.jpg`.

## Conclusões e Próximos Passos
- O ETL está funcional e produz tiles consistentes; alinhamento e tiling demonstrados pelas figuras.
- O modelo (DPT-ViT) está integrado e treinável; o fluxo de inferência salva métricas e visualizações automaticamente.
- Próximos passos sugeridos: aumentar cobertura de testes para trechos de treinamento/inferência, registrar métricas quantitativas de fill em múltiplos pares de teste e adicionar comparativos com métodos base (ex.: interpolação clássica).
