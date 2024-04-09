import pandas as pd

df = pd.read_csv('sub_sample_dataset2.csv')
id_to_title = {k:v for k, v in zip(df['citation.paperId'].to_list(), df['citation.title'].to_list())} | {k:v for k, v in zip(df['model.paperId'].to_list(), df['model.title'].to_list())}


def implies(a: bool, b: bool) -> bool:
    return not(a) or b

def title_from_path(path:str):
        id = path.split('/')[-1].replace('.mmd','')
        print(id)
        return id_to_title.get(id)