# MemeInterpret: Towards an All-in-one Dataset for Meme Understanding

Data and code accompanying the paper "MemeInterpret: Towards an All-in-one Dataset for Meme Understanding" (to appear at EMNLP Findings 2025)

## Data

Download images from the [Facebook Hateful Memes Challenge](https://ai.meta.com/tools/hatefulmemes/) into `fhm_image/original/`. If you cannot obtain the official dataset, there is an _unofficial_ dataset hosted on [huggingface](https://huggingface.co/datasets/neuralcatcher/hateful_memes/tree/main/img). Use at your own risk.

<!-- Run this script to prepare data for HatReD:
```bash
mkdir -p data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_test_reasonings.jsonl -P data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_train_reasonings.jsonl -P data/hatred
``` -->

## Install

```bash
pip install poetry
poetry shell
poetry install
git clone https://github.com/chjwon/LLM_EVAL
```

## Experiments

To be updated

## Cite

To be updated with EMNLP Findings 2025 citation.
