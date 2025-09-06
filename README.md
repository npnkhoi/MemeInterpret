# MemeInterpret: Towards an All-in-one Dataset for Meme Understanding

## Data

Download images from the [Facebook Hateful Memes Challenge](https://ai.meta.com/tools/hatefulmemes/) into `fhm_image/original/`.

<!-- Run this script to prepare data for HatReD:
```bash
mkdir -p data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_test_reasonings.jsonl -P data/hatred
wget https://github.com/Social-AI-Studio/HatReD/raw/refs/heads/main/datasets/hatred/annotations/fhm_train_reasonings.jsonl -P data/hatred
``` -->

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

## Note

It looks like the original FHM Dataset is not officially available anymore. There is an _unofficial_ dataset hosted at https://huggingface.co/datasets/neuralcatcher/hateful_memes/tree/main/img. Use at your own risk.