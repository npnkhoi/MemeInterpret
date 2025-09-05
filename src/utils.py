from enum import Enum
import random
from typing import Dict, List, Tuple
from src.prompt_arxiv import mllm_instruction_gen_SM, mllm_instruction_gen_BK
from PIL import Image
from transformers import PreTrainedTokenizerFast

###################### Prompt ######################

class PromptId(Enum):
    # General syntax: {task}/{sub_prompt_id}

    # Meme Captioning
    MC_NONE = "MC/none"
    MC_SM = "MC/SM"
    MC_BK = "MC/BK"
    MC_SM_BK = "MC/SM_BK"
    MC_AUTOSM_AUTOBK = "MC/autosm_autobk"
    MC_AUTOIC_TEXT = "MC/autoic_text"

    # Hateful Memes Detection
    HMD_MC = "HMD/MC"
    HMD_SM_BK_MC = "HMD/SM_BK_MC"
    HMD_AUTOSM_AUTOBK_AUTOMC = "HMD/autosm_autobk_automc"
    HMD_AUTOMC = "HMD/automc"
    HMD_NONE = "HMD/none"
    HMD_LLM_SM_BK_MC = "HMD/LLM_SM_BK_MC"

    # Image Captioning
    IC_NONE = "IC/none"

    # SM generation
    SM = "SM/none"

    # BK generation
    BK = "BK/none"

    # Explanation
    EXPL_SM_BK_MC = "EXPL/SM_BK_MC"
    EXPL_AUTOSM_AUTOBK_AUTOMC = "EXPL/autosm_autobk_automc"
    EXPL_LLM_SM_BK_MC = "EXPL/LLM_SM_BK_MC"

    # Joint tasks
    SMBKMC = "SM_BK_MC/none"

class PromptGenerator:
    LLAVA_SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
    PROMPTS = {
        PromptId.MC_NONE.value: ' '.join([
            "You will be provided with a meme.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single short sentence.",
            "The final message of this meme is:"
        ]),
        PromptId.MC_SM.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, and the background knowledge that a reader of the meme needs to possess before they can understand the message.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single sentence.",
            "\n### Description of its text and image: {}.",
            "\n### Message:"
        ]),
        PromptId.MC_BK.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, and the background knowledge that a reader of the meme needs to possess before they can understand the message.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single sentence.",
            "\n### Background knowledge: {}." +
            "\n### Message:"
        ]),
        PromptId.MC_SM_BK.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, and the background knowledge that a reader of the meme needs to possess before they can understand the message.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single sentence.",
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message:"
        ]),
        PromptId.MC_AUTOSM_AUTOBK.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, and the background knowledge that a reader of the meme needs to possess before they can understand the message.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single sentence.",
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message:"
        ]),
        PromptId.MC_AUTOIC_TEXT.value: ' '.join([
            "You will be provided with a meme, a description of its image, and the text written on it.",
            "Your task is to infer the message that the author is trying to convey through the meme.",
            "The message must be in one single sentence.",
            "\n### Description of the image: {}.",
            "\n### Text on the meme: {}.",
            "\n### Message:"
        ]),
        PromptId.HMD_MC.value: ' '.join([
            "You will be provided with a meme and the message that the author is trying to convey through the meme.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\nMessage: {}",
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.HMD_SM_BK_MC.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, and the background knowledge that a reader of the meme needs to possess before they can understand the message.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}",
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.HMD_LLM_SM_BK_MC.value: ' '.join([
            "You will be provided with a meme via its description of its text and image, the background knowledge that a reader of the meme needs to possess before they can understand the message, and the message conveyed by the meme.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}",
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.HMD_AUTOSM_AUTOBK_AUTOMC.value: ' '.join([
            "You will be provided with a meme via its description of its text and image, the background knowledge that a reader of the meme needs to possess before they can understand the message, and the message conveyed by the meme.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}",
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.HMD_AUTOMC.value: ' '.join([
            "You will be provided with a meme and the message that the author is trying to convey through the meme.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\nMessage: {}",
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.HMD_NONE.value: ' '.join([
            "You will be provided with a meme.",
            "Your task is to determine whether the meme is hateful or not.",
            "Format your answer as a binary classification, where 1 indicates that the meme is hateful and 0 indicates that the meme is not hateful.",
            "Only return the number."
            "\nIs this meme hateful? (0/1)"
        ]),
        PromptId.IC_NONE.value: (
            "Describe the image in a single sentence, emphasizing the exact names of all celebrities depicted."
        ),
        PromptId.SM.value: ' '.join([
            mllm_instruction_gen_SM,
            "\n### Surface message:"
        ]),
        PromptId.BK.value: ' '.join([
            mllm_instruction_gen_BK,
            "\n### Background knowledge:"
        ]),
        PromptId.EXPL_SM_BK_MC.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, the background knowledge that a reader of the meme needs to possess before they can understand the message, and the message conveyed by the meme.",
            "Your task is to explain why the meme is hateful.",
            "The explanation must be in one of the form (i) '<verb> <target> <predicate>' or (ii) 'use of derogatory terms against <target> <predicate>', where <target> represents the attacked social target and <predicate> highlights the hateful implication.",
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}" +
            "\n### Explanation:"
        ]),
        PromptId.EXPL_AUTOSM_AUTOBK_AUTOMC.value: ' '.join([
            "You will be provided with a meme, a description of its text and image, the background knowledge that a reader of the meme needs to possess before they can understand the message, and the message conveyed by the meme.",
            "Your task is to explain why the meme is hateful.",
            "The explanation must be in one of the form (i) '<verb> <target> <predicate>' or (ii) 'use of derogatory terms against <target> <predicate>', where <target> represents the attacked social target and <predicate> highlights the hateful implication.",
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}" +
            "\n### Explanation:"
        ]),
        PromptId.EXPL_LLM_SM_BK_MC.value: ' '.join([
            "You will be provided with a meme via the description of its text and image, the background knowledge that a reader of the meme needs to possess before they can understand the message, and the message conveyed by the meme.",
            "Your task is to explain why the meme is hateful.",
            "The explanation must be in one of the form (i) <verb> <target> <predicate> or (ii) use of derogatory terms against <target> <predicate>, where <target> represents the attacked social target and <predicate> highlights the hateful implication.",
            "\n### Description of its text and image: {}.",
            "\n### Background knowledge: {}." +
            "\n### Message: {}" +
            "\n### Explanation:"
        ]),
        PromptId.SMBKMC.value: ' '.join([
            "You will be provided with a meme.",
            "Generate a surface message (SM), background knowledge (BK), and a meme caption (MC) for the meme.",
            "A surface message is defined as \"what the meme is saying directly, including any text, images, or symbols present, and excluding interpretation of deeper meaning.\"",
            "A background knowledge list is defined as \"the minimum list of factual statements that is missing from the meme. It is the knowledge that needs to be combined with visual and textual cues from the meme in order to understand the meme's meaning.\"",
            "A meme caption is defined as \"the message that the author is trying to convey through the meme, written in one single short sentence.\"",
            "Format your answers as \"<SM_start> {Your SM} <SM_end> <BK_start> {Your BK} <BK_end> <MC_start> {Your MC} <MC_end>\""
        ])
    }

    @staticmethod
    def get_user_prompt(prompt_id: str, example: Dict) -> str:
        prompt = PromptGenerator.PROMPTS[prompt_id]
        
        # Substitute placeholders
        if prompt_id == PromptId.MC_SM.value:
            prompt = prompt.format(example["surface_message"])
        elif prompt_id == PromptId.MC_BK.value:
            prompt = prompt.format(' '.join(example["background_knowledge_list"]))
        elif prompt_id == PromptId.MC_SM_BK.value:
            prompt = prompt.format(
                example["surface_message"], 
                ' '.join(example["background_knowledge_list"])
            )
        elif prompt_id == PromptId.HMD_MC.value:
            prompt = prompt.format(example["meme_caption_list"][0])
        elif prompt_id in [
            PromptId.HMD_SM_BK_MC.value, 
            PromptId.HMD_LLM_SM_BK_MC.value,
            PromptId.EXPL_SM_BK_MC.value, 
            PromptId.EXPL_LLM_SM_BK_MC.value,
        ]:
            prompt = prompt.format(
                example["surface_message"], 
                ' '.join(example["background_knowledge_list"]), 
                example["meme_caption_list"][0]
            )
        elif prompt_id == PromptId.MC_AUTOIC_TEXT.value:
            prompt = prompt.format(example["auto_image_caption"], example["text"])
        elif prompt_id == PromptId.MC_AUTOSM_AUTOBK.value:
            prompt = prompt.format(
                example["auto_surface_message"], 
                example["auto_background_knowledge"]
            )
        elif prompt_id in [PromptId.HMD_AUTOSM_AUTOBK_AUTOMC.value, PromptId.EXPL_AUTOSM_AUTOBK_AUTOMC.value]:
            prompt = prompt.format(
                example["auto_surface_message"], 
                example["auto_background_knowledge"], 
                example["auto_meme_caption"]
            )
        elif prompt_id == PromptId.HMD_AUTOMC.value:
            prompt = prompt.format(example["auto_meme_caption"])
        
        assert '{}' not in prompt, f"Prompt {prompt_id} still has placeholders: {prompt}"

        return prompt
    
###################### Collators ######################

class Collator:
    def __init__(self, processor, tokenizer, prompt_id: str) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt_id = prompt_id
        self.task = prompt_id.split('/')[0]
        self.is_llama = isinstance(processor, PreTrainedTokenizerFast)
        if self.is_llama:
            # llama does not have pad_token, so we set it to eos_token, should be fine
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_ground_truth(self, task, example):
        if task == 'HMD':
            return str(example['label'])
        elif task == 'MC':
            return example['meme_caption_list'][0]
        elif task == 'IC':
            return None # No ground truth for image captioning
        elif task == 'SM':
            return example['surface_message']
        elif task == 'BK':
            return ' '.join(example['background_knowledge_list'])
        elif task == 'EXPL':
            return example['explanation']
        elif task == 'SM_BK_MC':
            return ' '.join([
                f"<SM_start> {example['surface_message']} <SM_end>",
                f"<BK_start> {' '.join(example['background_knowledge_list'])} <BK_end>",
                f"<MC_start> {example['meme_caption_list'][0]} <MC_end>"
            ])
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _get_basic_batch(self, examples, is_train: bool) -> Tuple[Dict, List[str]]:
        """
        Returns a basic batch (Dict) and a list of ground truths.

        The basic batch is a dictionary with keys:
        - input_ids: if training, containing the ground truth
        - attention_mask
        - pixel_values (only for llava)
        """
        images = []
        texts = []
        ground_truths = []
        for example in examples:
            ground_truth = self.get_ground_truth(self.task, example)
            
            # Get the prompt depending on the model
            if self.is_llama:
                # conversation = [
                #     {"role": "system", "content": "You are a helpful assistant."},
                #     {"role": "user", "content": f"<image>\n{PromptGenerator.get_user_prompt(self.prompt_id, example)}"},
                #     {"role": "assistant", "content": ground_truth if is_train else ""}
                # ]
                # prompt = self.tokenizer.apply_chat_template(conversation, return_tensors='pt', tokenize=False)
                
                # Use the same prompt template as Llava because we are fine-tuning anyway
                prompt = (
                    f"{PromptGenerator.LLAVA_SYSTEM_PROMPT} " + 
                    f"USER: <image>\n{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" +
                    f"ASSISTANT: {ground_truth if is_train else ''}"
                )
            else:
                image = Image.open(example['img_path']).convert('RGB')
                images.append(image)
                prompt = (
                    f"{PromptGenerator.LLAVA_SYSTEM_PROMPT} " + 
                    f"USER: <image>\n{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" + 
                    f"ASSISTANT: {ground_truth if is_train else ''}"
                )
            
            texts.append(prompt)
            ground_truths.append(ground_truth)
        
        if self.is_llama:
            batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        else:
            batch = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")

        return batch, ground_truths
    
    def train_collate_fn(self, examples):
        batch, _ = self._get_basic_batch(examples, is_train=True)

        # Add `labels` field to the batch
        # copied from tutorials, but not sure why inputs and and labels are the same
        # maybe internally they do shifting to train on next tok predicion already?
        # => Yes
        labels = batch["input_ids"].clone() 
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    def eval_collate_fn(self, examples):
        batch, answers = self._get_basic_batch(examples, is_train=False)
        
        # Add `answers` and `image_paths` fields to the batch
        image_paths = [example['img_path'] for example in examples]
        batch['answers'] = answers
        batch['image_paths'] = image_paths

        return batch
    
###################### Others ######################
def get_metric_names(task, mode):
    assert task in ['MC', 'HMD', 'IC', 'SM', 'BK', 'EXPL', 'SM_BK_MC']
    assert mode in ['train', 'eval']
    # WARNING: must put eval metric AT THE END!
    if task in ['MC', 'SM', 'BK', 'EXPL', 'SM_BK_MC']:
        if mode == 'train':
            return ['nli']
        else:
            return ['bleu', 'rouge', 'bertscore', 'nli']
    else:
        return ['acc']

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'

class SubsetEnum(Enum):
    ALL = "all"
    HALF = "half"
    PILOT = "pilot"
