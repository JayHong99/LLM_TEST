import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

    
BASE_MODEL = 'decapoda-research/llama-7b-hf'
model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit = False, torch_dtype = torch.float16, device_map = "cuda")
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = (0)
tokenizer.padding_side = 'left'
CUTOFF_LEN = 512
    
def won_to_name(score : float) : 
    if score > 0 : 
        return "First party won the case"
    else : 
        return "Second party won the case"

def convert_df_to_data(df : pd.DataFrame, phase) : 
    dataset_data = []
    for idx, row in df.iterrows() : 
        row_data = {
                    'instruction' : f'Court Cases : Guess who won the case, first party {row["first_party"]} or second party {row["second_party"]}\n',
                    'input' : row['facts'],
                    'output' : won_to_name(row['first_party_winner']) if phase == 'train' else None,
                    }
        dataset_data.append(row_data)
    return dataset_data




def generate_prompt(data_point) : 
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

def tokenize(prompt, CUTOFF_LEN = 512, add_eos_token = True) : 
    result = tokenizer(
                        prompt,
                        truncation=True,
                        max_length=CUTOFF_LEN,
                        padding = False,
                        return_tensors=True   
    )
    if (
        result['input_ids'][-1] != tokenizer.eos_token_id
        and len(result['input_ids']) < CUTOFF_LEN
        and add_eos_token
    ):
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    
    result['labels'] = result['input_ids'].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

