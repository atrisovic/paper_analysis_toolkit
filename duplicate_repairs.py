from pydantic import BaseModel, confloat
from llama_cpp import Llama
from src.language_models.ChatInterface import LlamaCPPChatInterface
from src.language_models.QuestionSet import QuestionSet
from datetime import datetime
from src.prompts.disambiguation_prompts import PROMPT1
import pandas as pd
from tqdm import tqdm
import argparse
from src.process.Cluster import Cluster
import json
from csv import DictWriter
import regex as re
import os


class StringAnswer(BaseModel):
    answer: str


username = os.getenv('USER') or os.getenv('USERNAME')

    
##### ARGUMENT PARSING #####

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of foundation models analyzed.')
parser.add_argument('-s', '--seed', default = 0, type = int, help = "Seed used for all random processes. Default is 0.")

args = parser.parse_args()
cluster = Cluster(index = args.index, worker_count = args.workers, limit = args.limit, seed = args.seed)


##### LOAD MODEL AND QUESTION SET #####

model_path = 'saved_models/models--bullerwins--Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'
model = Llama(model_path, n_gpu_layers = -1, n_ctx = 4096, verbose = False)
interface = LlamaCPPChatInterface(model = model, outputClass = StringAnswer)


mapping_path = '/data1/groups/futuretech/atrisovic/osfm/paper_analysis_toolkit/notebooks/master_dup_mapping.txt'
with open(mapping_path, 'r') as f:
    duplicate_mapping = json.load(f)


samples_path = '/data1/groups/futuretech/atrisovic/osfm/saved_results/classifier/trials/results/premicrosoft_meeting_classified.csv'
df = cluster.clusterDataframe(pd.read_csv(samples_path))


df = df[df['modelId'].apply(lambda id: id in duplicate_mapping)]
df = df[df['classification'].apply(lambda c: c in {'extends', 'uses'})]


df.drop_duplicates(subset = ['modelId', 'paperId', 'multisentence'], inplace = True)

output_path = '/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/premicrosoft_meeting_results_disambiguated.csv'
with open(output_path, 'a') as f:
        if f.tell() == 0:
            csv_writer = DictWriter(f, fieldnames=df.columns)
            csv_writer.writeheader()



##### ASK QUESTIONS! #####


for idx, row in tqdm(df.iterrows(), total = len(df)):
    model_name, variant_list = duplicate_mapping[row['modelId']]
    
    piped_variant = '" | "'.join(variant_list)
    comma_variant = '"' + '", "'.join(variant_list) + '"'
    prompt = PROMPT1.format(foundation_model_name = model_name, 
                            piped_variant = piped_variant,
                            comma_variant = comma_variant, 
                            sentences = row['multisentence'])

    
    response = interface.generateAsModel(input = prompt)
    
    row['modelKey'] = response.answer
    df['modelKey'].at[idx] = response.answer
    
    with open(output_path, 'a') as f:
        csv_writer = DictWriter(f, fieldnames=df.columns)
        csv_writer.writerow(dict(row))
    
distribution = df['modelKey'].apply(lambda s: s if s in {'UNCLEAR', 'ALL'} else 'SPECIFIED').value_counts()/len(df)
print(distribution)