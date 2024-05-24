import pandas as pd
from typing import Tuple
from math import ceil
import random, json
from os.path import basename
from torch import backends, cuda

df = pd.read_csv('./data/sub_sample_dataset2.csv')
id_to_title = {k:v for k, v in zip(df['citation.paperId'].to_list(), df['citation.title'].to_list())} | {k:v for k, v in zip(df['model.paperId'].to_list(), df['model.title'].to_list())}


def stemmed_basename(path: str):
    return basename(path).split('.')[0]

def implies(a: bool, b: bool) -> bool:
    return not(a) or b

def title_from_path(path:str):
        id = path.split('/')[-1].replace('.mmd','')
        return id_to_title.get(id)

def extract_paper_metadata(path: str):
    paper_years = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            if (not json_line.get('error')):
                paper_years[json_line['paperId']] = json_line['year']
    return paper_years

def get_device():
    return 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

def load_affiliations_results(path: str):
    with open(path, 'r') as f: 
        dicts = map(json.loads, f.readlines())
        return {k:v for d in dicts for k, v in d.items()}
    
def merge_affiliation_results(*args):
    result_sets = list(map(load_affiliations_results, args))
    all_keys = {key for result in result_sets for key in result}
    
    
    chosen_values = {}
    for key in all_keys:
        dict_values = filter(lambda s: isinstance(s, dict), [result.get(key) for result in result_sets])
        string_values = filter(lambda s: isinstance(s, str), [result.get(key) for result in result_sets])

        chosen_values[key] = next(iter(dict_values), None) or next(iter(string_values), None)
        
    return chosen_values