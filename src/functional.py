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
    
def clusterOrLimitList(L: list, cluster_info: Tuple[int, int] = None, limit: int = None):
    if (not limit and not cluster_info):
        return L

    random.seed(0)
    random.shuffle(L) #need consistency accross jobs
    
    L = L[:limit or len(L)]
        
    if cluster_info:
        cluster_index, cluster_count = cluster_info
        chunk_length = ceil(len(L)/cluster_count)
    
        L = L[chunk_length * (cluster_index - 1): chunk_length * cluster_index] #cluster_index is one-indexed

    return L


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