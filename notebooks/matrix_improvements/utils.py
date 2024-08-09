import requests
import json
from typing import Optional
import ollama
import pandas as pd
import numpy as np
import hashlib
 


master_path = '~/Desktop/2. FutureTech/uniform_sample/raw/uniform_base_sample'


def prompt(input, model, temperature = 0.5, connection_on = False):
    if not connection_on:
        raise Exception
    
    api_key = 'these are not the droids you are looking for'

    api_url = 'https://api.openai.com/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    
    if model == 'ollama':
        response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': input,
            'temperature': temperature
        }, 
        ])
        return response['message']['content']

    else:
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': input},
            ],
            'max_tokens': 2000,
            'temperature': temperature,
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code}")
            return response.json()

def stripJSON(text: Optional[str], added_bracket = False, debug = False) -> Optional[dict]:
    if (text is None):
        raise Exception("Got empty text!")
    
    #text = text.replace('\(', '(').replace('\)', ')') #Mistral likes to add escape sequences, for some unknown reason
    text = text.replace("\\", "\\\\")
    text = text.replace("```", "").replace('json', '')
    text = text.replace("“", "\"")
    
    if (debug):
        print(text)
        try:
            json.loads(text)
        except Exception as e:
            print(e)

    # a very manual way of finding our JSON string within the output
    
    all_open_brackets = [i for i, ltr in enumerate(text) if ltr == '{']
    obj = None
    for start in all_open_brackets:
        balance_counter = 1
        for offset, chr in enumerate(text[start + 1:], start = 1):
            balance_counter += (1 if chr == '{' else -1 if chr == '}' else 0)
            if (balance_counter == 0):
                try:
                    obj = json.loads(text[start: start + offset + 1])
                except:
                    pass
                break
        if (obj is not None):
            break
        
    # sometimes we miss the first or last bracket (dumb LLM), so we add it manually.
    if not added_bracket:
        return (stripJSON('{' + text, added_bracket = True) or stripJSON(text + '}', added_bracket = True))
        
    return obj




def update_labels(path, 
                  column = 'alex2',
                  ground_truth_path = master_path + '.csv',
                  save = False):
    ground_df = pd.read_csv(ground_truth_path)
    to_update_df = pd.read_csv(path)
    to_update_df.drop(columns=set(f'Unnamed: 0.{i}' for i in range(0, 100)).intersection(set(to_update_df.columns)), inplace = True)
    
    updated = 0
    for idx in to_update_df.index:
        mask = ground_df['multisentence'] == to_update_df['multisentence'].loc[idx]
        assert(mask.sum() == 1)
        
        if to_update_df[column].loc[idx] != ground_df[column].loc[idx]:
            updated += 1
            to_update_df[column].loc[idx] = ground_df[column].loc[idx]
    
    print(f"Updated {updated} labels in {path}.")
    
    if (save):
        print(f"Saving to original path. ")
        to_update_df.to_csv(path, index = False)
    return to_update_df


def get_answer_vector(r_, length, verbose = False):
    if isinstance(r_, str):
        r = stripJSON(r_)
        
    if r is None or r is np.nan:
        if (verbose):
            print(r_)
        return None
    

    for key, item in r.items():
        if key == 'error':
            return None
        r[key] = {'true': True, 'false': False, None: False}.get(item.lower() if isinstance(item, str) else None) or False
        
    answer_vector = np.vectorize(lambda s: 1 if s else 0)(np.array([r.get(f"answer_{i}") for i in range(1, length + 1)]))
    return answer_vector

def hash_dataframe(df: pd.DataFrame):
    return hashlib.md5(bytes(str(df), encoding='utf-8')).digest()
