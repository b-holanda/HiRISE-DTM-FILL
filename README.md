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


conda create -n marsfill-env -c conda-forge python=3.11 gdal numpy

conda activate marsfill-env

conda install -c conda-forge libgdal-jp2openjpeg

pip install -e .
```

## Uso

```bash
marsfill --help
```

### Gerar dataset usando 100 pares DTM+ORTHO da Hirise

```bash
marsfill dataset build
```

```bash
marsfill model train
```

```bash
marsfill model test
```

```bash
marsfill fill --source /seu/dtm.IMG --out /caminho/saida/dtm_filled.IMG
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
