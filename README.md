# marsfill

-----

## Instalação

```bash
# 3. Clone este repositório
git clone https://github.com/b-holanda/HiRISE-DTM-FILL.git

cd HiRISE-DTM-FILL

python -m venv .venv

source ./.venv/bin/actvate

pip install -e .
```

## Uso

```bash
marsfill --help
```

```bash
marsfill dataset:build
```

```bash
marsfill model:train
```

```bash
marsfill model:test
```

```bash
marsfill model:load
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
