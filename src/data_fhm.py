import json
import os
import warnings
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import ast

FHM_ORIG_IMG_DIR = 'data/fhm_img/original/'
FHM_NOTEXT_IMG_DIR = 'data/fhm_img/notext/'

class FastFHMDataset(Dataset):
    def __init__(self, data_list: list, load_img: bool=False):
        self.data_list = data_list
        self.load_img = load_img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        item['img_path'] = os.path.join(FHM_ORIG_IMG_DIR, item['img_path'].split('/')[-1])
        # add autobk and autosm
        if self.load_img:
            item['img'] = Image.open(item['img_path']).convert('RGB')
        return item

train_list = json.load(open('data/memeinterpret/train_data.json'))
dev_list = json.load(open('data/memeinterpret/dev_data.json'))
test_list = json.load(open('data/memeinterpret/test_data.json'))

train_dataset = FastFHMDataset(train_list)
dev_dataset = FastFHMDataset(dev_list)
test_dataset = FastFHMDataset(test_list)
traindev_dataset = FastFHMDataset(train_list + dev_list)
devtest_dataset = FastFHMDataset(dev_list + test_list)

###################### explanation ######################
hatred_train_fn = "data/hatred/fhm_train_reasonings.jsonl"
hatred_test_fn = "data/hatred/fhm_test_reasonings.jsonl"
hatred_test_images = []

# explanation_dict: img_filename -> explanation
explanation_dict = {}
with open(hatred_train_fn, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        explanation_dict[item['img']] = item['reasonings'][0]

with open(hatred_test_fn, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        explanation_dict[item['img']] = item['reasonings'][0]
        hatred_test_images.append(item['img'])

# use the pre-split images to split the *intersection* between hatred and memeinterpret (train list)
hatred_splits = json.load(open("data/hatred/hatred_split_v2.json"))
# print(f"size of train+dev (fhm): {len(train_list) + len(dev_list)}")
hatred_train_list = [
    x for x in train_list + dev_list
    if x['img_path'].split('/')[-1] in hatred_splits['train']
]
hatred_dev_list = [
    x for x in train_list + dev_list
    if x['img_path'].split('/')[-1] in hatred_splits['dev']
]
hatred_test_list = [
    x for x in train_list + dev_list
    if x['img_path'].split('/')[-1] in hatred_splits['test']
] 

# print("total expl datasets size:", len(hatred_train_list) + len(hatred_dev_list) + len(hatred_test_list))

class ExplainDataset(FastFHMDataset):
    """
    Just like FHMDataset, but items also have an "explanation" field
    """
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img_fn = item['img_path'].split('/')[-1]
        item['explanation'] = explanation_dict[img_fn]
        return item

expl_train_dataset = ExplainDataset(hatred_train_list)
expl_dev_dataset = ExplainDataset(hatred_dev_list)
expl_test_dataset = ExplainDataset(hatred_test_list)
# print("total expl Datasets size:", len(expl_train_dataset) + len(expl_dev_dataset) + len(expl_test_dataset))


