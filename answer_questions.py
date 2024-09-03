from pydantic import BaseModel, confloat
from llama_cpp import Llama
from src.language_models.ChatInterface import LlamaCPPChatInterface
from src.language_models.QuestionSet import QuestionSet
from datetime import datetime
from src.prompts.citation_prompts import SINGLEPROMPT, SINGLEPROMPT_COT, SINGLEPROMPT_COT_CONFIDENCE
from src.prompts.questions_sets import *
import pandas as pd
from tqdm import tqdm
import argparse
from src.process.Cluster import Cluster
import json
from csv import DictWriter
import regex as re
import os

username = os.getenv('USER') or os.getenv('USERNAME')

question_list = weighted


class BoolAnswer(BaseModel):
    answer: bool
    
class COTAnswer(BaseModel):
    explanation: str
    answer: bool
    
class COTConfidenceAnswer(BaseModel):
    explanation: str
    answer: confloat(ge = 0, le = 1) # type: ignore
    
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

question_set = QuestionSet(questions = question_list)
prompt = SINGLEPROMPT_COT
model = Llama(model_path, n_gpu_layers = -1, n_ctx = 4096, verbose = False)
interface = LlamaCPPChatInterface(model = model, outputClass = COTAnswer)


#'/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/labeled_samples_4_of_32_2024-08-14 17:28:00.csv'
#'/home/gridsan/atrisovic/futuretech_shared/atrisovic/osfm/saved_results/classifier/trials/results/labeled_samples_15_of_32_2024-08-14 20:38:00.csv'
#'/home/gridsan/atrisovic/futuretech_shared/atrisovic/osfm/saved_results/classifier/trials/results/labeled_samples_14_of_32_2024-08-14 20:38:00.csv' #None

#csv_path = '/home/gridsan/afogelson/osfm/saved_results/classifier/labeled_samples.csv' 
csv_path = f'/home/gridsan/{username}/osfm/saved_results/citations/0814/results_2024-08-14_removed_double_reject_fms.csv'
df = cluster.clusterDataframe(pd.read_csv(csv_path))

df.rename(columns={'sentence':'multisentence'}, inplace = True)
df['strippedModelKey'] = df['modelKey'].apply(lambda s: re.sub('^\d+\_', '', s))

vector_column, answer_column = 'answer_vector', 'answer_string'
if (vector_column not in df.columns):
    df[vector_column] = [None for i in range(len(df))]
    df[answer_column] = [None for i in range(len(df))]

##### SAVE METADATA #####
right_now = datetime.now().replace(microsecond=0, second=0)

worker_label = f'{args.index}_of_{args.workers}_' if args.workers > 1 else ''
output_path = f'/home/gridsan/{username}/osfm/saved_results/classifier/trials/results/premicrosoft_meeting_results.csv'
#output_path = f'/home/gridsan/{username}/osfm/saved_results/classifier/trials/results/sample_results.csv'

metadata_path = f'/home/gridsan/{username}/osfm/saved_results/classifier/trials/metadata/premicrosoft_meeting_results.txt' 
#metadata_path = f'/home/gridsan/{username}/osfm/saved_results/classifier/trials/metadata/sample_results.txt' 


existing_results_path = output_path #'/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/merged_classification_results.csv' #output_path


##### LOAD CLUSTERED DATAFRAME #####


if (args.workers == 1 or args.index == 1):
    with open(metadata_path, 'a') as f:
        f.write(f"PROMPT:\n\n{prompt}\n")
        all_questions = '\n'.join(question_set.questions)
        f.write(f"QUESTION SET:\n\n{all_questions}")


##### ASK QUESTIONS! #####

with open(output_path, 'a') as f:
        if f.tell() == 0:
            csv_writer = DictWriter(f, fieldnames=df.columns)
            csv_writer.writeheader()

skip_primary_keys = {} if existing_results_path is None else {(row['modelKey'], row['paperId'], row['multisentence']) 
                     for idx, row in pd.read_csv(existing_results_path).iterrows()}
df = df[df.apply(lambda row: (row['modelKey'], row['paperId'], row['multisentence']) not in skip_primary_keys, axis = 1)]

for idx, row in tqdm(list(df.iterrows())):
    multisentence, modelKey = row['multisentence'], row['strippedModelKey']
    
    answers = question_set.ask_questions(multisentence, metadata=modelKey, chat_interface=interface, prompt = prompt)
    vector = question_set.get_answer_vector(response = answers, verbose = True)

    #row[answer_column] = json.dumps([None if not answer else answer.model_dump() for answer in answers])
    row[vector_column] = vector
    
    with open(output_path, 'a') as f:
        csv_writer = DictWriter(f, fieldnames=df.columns)
        csv_writer.writerow(dict(row))
    