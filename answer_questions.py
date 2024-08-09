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


##### LOAD CLUSTERED DATAFRAME #####

csv_path = '/home/gridsan/afogelson/osfm/saved_results/classifier/labeled_samples.csv' 
#csv_path = '/home/gridsan/afogelson/osfm/saved_results/citations/0707/all_sentences.csv'
df = cluster.clusterDataframe(pd.read_csv(csv_path))
vector_column, answer_column = 'answer_vector', 'answer_string'
if (vector_column not in df.columns):
    df[vector_column] = [None for i in range(len(df))]
    df[answer_column] = [None for i in range(len(df))]

##### SAVE METADATA #####
right_now = datetime.now().replace(microsecond=0, second=0)

worker_label = f'{args.index}_of_{args.workers}_' if args.workers > 1 else ''
new_path = f'/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/labeled_samples_{worker_label}{right_now}.csv' 
metadata_path = f'/home/gridsan/afogelson/osfm/saved_results/classifier/trials/metadata/labeled_samples_{worker_label}{right_now}.txt' 

if (args.workers == 1 or args.index == 1):
    with open(metadata_path, 'w') as f:
        f.write(f"PROMPT:\n\n{prompt}\n")
        all_questions = '\n'.join(question_set.questions)
        f.write(f"QUESTION SET:\n\n{all_questions}")


##### ASK QUESTIONS! #####

for idx, row in tqdm(list(df.iterrows())):
    if (row[vector_column] is not None): #don't overwrite existing answers
        continue

    multisentence, modelKey = row['multisentence'], row['strippedModelKey']
    answers = question_set.ask_questions(multisentence, metadata=modelKey, chat_interface=interface, prompt = prompt)
    df[answer_column].at[idx] = json.dumps([None if not answer else answer.model_dump() for answer in answers])
    vector = question_set.get_answer_vector(response = answers, verbose = True)
    df[vector_column].at[idx] = vector

    
    df.to_csv(new_path, index = False)
    