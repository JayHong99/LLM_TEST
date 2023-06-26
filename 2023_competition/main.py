import json
import pandas as pd
import argparse
from pathlib import Path

import transformers
import textwrap
import os
import sys
from typing import List
from datasets import load_dataset
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import fire
import torch
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
 
from utils import convert_df_to_data
from utils import generate_and_tokenize_prompt


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--json_root', type=str, default='data')
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--cutoff_len', type=int, default=512)

def create_json(data_root, json_root, data_name) : 
    df = pd.read_csv(data_root / f'{data_name}.csv')
    data = convert_df_to_data(df, data_name)
    with open(json_root / f'{data_name}.json', 'w') as f : 
        json.dump(data, f)



if __name__ == "__main__" : 
    args = parser.parse_args()
    DEVICE = "cuda" if args.device == 'gpu' else 'cpu'

    
    data_root = Path(args.data_root)
    json_root = Path(args.json_root)
    create_json(data_root, json_root, 'train')
    create_json(data_root, json_root, 'test')

    
    train_data = load_dataset('json', data_files = json_root / 'train.json', split = 'train')
    train_val = data['train'].train_test_split(test_size = 0.1, shuffle = True, random_state = 42)
    train_data = train_val['train'].map(generate_and_tokenize_prompt, batched = True)
    valid_data = train_val['test'].map(generate_and_tokenize_prompt, batched = True)
    test_data  = load_dataset('json', data_files = json_root / 'test.json', split = 'train').map(generate_and_tokenize_prompt, batched = True)
    
    print(train_data)