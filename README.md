# MemeInterpret: Towards an All-in-one Dataset for Meme Understanding

## Data

Download FHM images from into `fhm_image/original/`.

Run this script to prepare data for HatReD:
```bash
mkdir -p data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_test_reasonings.jsonl -P data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_train_reasonings.jsonl -P data/hatred
```

## Install

Required pip pre-installed.
```bash
pip install poetry
poetry shell
poetry install
git clone https://github.com/chjwon/LLM_EVAL
```

## Experiments

Table 3
