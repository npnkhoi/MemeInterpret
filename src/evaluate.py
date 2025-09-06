"""
Do both inference and evaluation on the test set
"""

import json
import os

import numpy as np
from typing import Dict
from LLM_EVAL.utils import bert_score, bleu_score, rougue_score, selfcheck_nli_score
from src.utils import Collator, Color, PromptId, get_metric_names
# TODO: update the explain train and test dataset
from src.data_fhm import (
    test_dataset, train_dataset, traindev_dataset,
    expl_test_dataset
)
from transformers import (
    LlavaForConditionalGeneration,
    LlamaForCausalLM,
    AutoProcessor,
    LlavaProcessor
)
from PIL import Image
from torch.utils.data import DataLoader, Subset
import argparse
import wandb
from tqdm import tqdm
import torch
import warnings


def evaluate(model, processor, dataloader, metrics, max_new_tokens: int, evaluate=True) -> Dict[str, float]:
    # show the first batch
    batch = next(iter(dataloader))
    print('A DEV BATCH:\n', batch.keys())
    print(f"{Color.GREEN}[Decoded input_ids]{Color.END}:\n```\n{processor.decode(batch['input_ids'][0])}\n```")
    print(f'{Color.GREEN}[Answer]{Color.END}:', batch['answers'][0])

    model.eval()
    with torch.no_grad():
        # Get all the texts first
        all_generated_texts = []
        all_answers = []
        all_image_paths = []
        for batch in tqdm(dataloader, desc="Generating texts"):
            # note: answers is just a list[str]
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if 'pixel_values' in batch:
                outputs = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    max_new_tokens=max_new_tokens
                )
            else:
                outputs = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_new_tokens
                )
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            all_generated_texts.extend(generated_texts)
            all_answers.extend(batch['answers'])
            all_image_paths.extend(batch['image_paths'])

        # Now score
        ret = {
            'overall': {},
            'per_image': {}
        }
        print('\nScoring...')
        for image_path, generated_text, answer in tqdm(zip(all_image_paths, all_generated_texts, all_answers), total=len(all_image_paths)):
            if 'ASSISTANT: ' in generated_text:
                trimmed_generated_text = generated_text.split('ASSISTANT: ')[1]
            elif 'assistant\n\n' in generated_text:
                trimmed_generated_text = generated_text.split('assistant\n\n')[1].strip()
            else:
                trimmed_generated_text = 'N/A'

            ret['per_image'][image_path] = {
                'full_output': generated_text,
                'trimmed_output': trimmed_generated_text,
                'answer': answer,
            }
            
            if 'bleu' in metrics:
                ret['per_image'][image_path]['bleu'] = bleu_score(trimmed_generated_text, answer)
            if 'rouge' in metrics:
                ret['per_image'][image_path]['rouge'] = rougue_score(trimmed_generated_text, answer)
            if 'bertscore' in metrics:
                ret['per_image'][image_path]['bertscore'] = bert_score(trimmed_generated_text, answer)
            if 'nli' in metrics:
                ret['per_image'][image_path]['nli'] = selfcheck_nli_score(trimmed_generated_text, answer)
            if 'acc' in metrics:
                ret['per_image'][image_path]['acc'] = int(trimmed_generated_text == answer)

    for metric in metrics:
        ret['overall'][metric] = np.mean([x[metric] for x in ret['per_image'].values()])

    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--peft_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prompt_id', type=str, default=PromptId.MC_NONE, choices=[memeber.value for memeber in PromptId])
    parser.add_argument('--run_name', type=str, default='llava_inference')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--dataset', choices=['interpret', 'hatred-main'], default='interpret')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--no_evaluate', action='store_true')
    args = parser.parse_args()

    wandb.init(
        project='arr-oct24-inference',
        name=args.run_name,
        entity='khoi-ml',
    )

    # Data
    task = args.prompt_id.split('/')[0]
    is_llama = 'llama' in args.base_model_id.lower()
    # processor = AutoProcessor.from_pretrained(args.base_model_id)
    # HACK, won't work for llama
    processor  = LlavaProcessor.from_pretrained(args.base_model_id)
    tokenizer = processor if is_llama else processor.tokenizer
    collator = Collator(processor, tokenizer, args.prompt_id)
    
    # choose a different dataset for explanation task
    if args.split == 'traindev':
        assert task != 'EXPL'
        dataset = traindev_dataset
    
    elif args.split == 'test':
        if args.dataset == 'hatred-main':
            dataset = expl_test_dataset
        else: # 'interpret'
            dataset = test_dataset
    
    else: # 'train'
        assert task != 'EXPL'
        dataset = train_dataset
    
    if args.pilot:
        print('Pilot mode')
        dataset = Subset(dataset, range(10)) # for testing
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # no need to shuffle for testing
        collate_fn=collator.eval_collate_fn
    )
    
    # Model
    if not is_llama:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.base_model_id, torch_dtype=torch.float16,
            device_map=args.device_map
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model_id, torch_dtype=torch.float16,
            device_map=args.device_map
        )

    if args.peft_id is not None:
        print('Loading adapter...')
        model.load_adapter(args.peft_id)
        # TODO: enable this!
        # training_args = json.load(open(f'{args.peft_id}/training_args.json'))
        # assert training_args['prompt_id'] == args.prompt_id
    
    # Inference and evaluation
    if args.no_evaluate:
        metrics = []
    else:
        metrics = get_metric_names(task, 'eval')
    results = evaluate(model, processor, dataloader, metrics, args.max_new_tokens)

    os.makedirs('output', exist_ok=True)
    output_file = f'output/{args.run_name}_{wandb.run.id}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'Output saved to {output_file}')

    wandb.finish()
