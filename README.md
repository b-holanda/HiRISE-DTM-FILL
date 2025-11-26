# marsfill

-----

## Instalação

```bash
git clone https://github.com/b-holanda/HiRISE-DTM-FILL.git

cd HiRISE-DTM-FILL

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sudo chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

source  ~/miniconda3/bin/activate

conda create -n marsfill-env -f environment.yml

conda activate marsfill-env

```

## Uso

### Gerar rotulos de treinamento usando 100 pares de datasets DTM+ORTHO da Hirise

---
É recomendado o seguinte hardware:
- **Memória RAM***: 32 GB
- **Núcleos de CPU***: 4
- **Espaço em disco**: 2 TB
---

```bash
./dataset.sh --profile dev --mode local
```

### Treinar modelo

---
O modelo base usado é o Intel/DPT-ViT-Large é recomando o seguinte hardware:
- **GPU**: 4 PLACAS NVIDIA A10G e 24 GB de memória
- **Memória de GPU**: 96GB
- **Memória RAM**: 192 GB
- **Núcleos de CPU***: 48
- **Espaço em disco**: 2 TB
---

```bash
./train.sh --profile dev --mode local
```

### Executar modelo

---
É recomando o seguinte hardware:
- **GPU**: 1 PLACA NVIDIA T4
- **Memória de GPU**: 24 GB
- **Memória RAM**: 16 GB
- **Núcleos de CPU***: 4
- **Espaço em disco**: 100 GB
---

```bash
./fill.sh -dtm /home/ubuntu/DTEPC_088676_2540_088162_2540_A01.IMG -ortho /home/ubuntu/ESP_088676_2540_RED_A_01_ORTHO.JP2
```

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
