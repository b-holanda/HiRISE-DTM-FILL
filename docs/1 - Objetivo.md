# Preenchimento de Lacunas (Gap-Filling) em DTMs CTX de Marte usando Inferência Monocular com Vision Transformers (ViT)

## Objetivo Geral

Treinar um modelo de IA (ViT-DPT) que aprenda a relação entre a textura/sombra de uma Ortoimagem (Entrada) e a forma do terreno (Saída). O modelo será então usado para inferir (gerar) dados de elevação plausíveis apenas para as áreas de lacuna (NoData) presentes nos DTMs oficiais, criando um produto de dados mais completo e utilizável.

## Objetivos Gerais

1. Criar um conjunto de dados limpo de pares (Ortoimagem, DTM) contendo apenas dados válidos, que servirá como "a verdade" para o treinamento do modelo.
2. Treinar o modelo ViT-DPT para prever com precisão um bloco de DTM válido quando apresentado a um bloco de ortoimagem.
3. Usar o modelo treinado para preencher as lacunas nos DTMs do conjunto de Teste e validar a qualidade desse preenchimento.
