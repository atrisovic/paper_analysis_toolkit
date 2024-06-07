from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import backends, cuda, bfloat16
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime 

from config import LLM_MODEL_NAME, LLM_MODEL_PATH, LLM_TOKENIZER_PATH

from src.process.Cluster import Cluster

from src.language_models.LLMModelSelector import LLMModelSelector
from src.prompts.affiliation_prompts import PROMPT3
from csv import DictWriter


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of documents scanned.')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('-s', '--seed', default = 0, type = int, help = "Seed used for all random processes. Default is 0.")

    args = parser.parse_args()
    right_now = datetime.now().replace(microsecond=0)
    
    cluster = Cluster(index = args.index, worker_count = args.workers, limit = args.limit, seed = args.seed)

        
    device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device = {device}")
    
    refresh = False
    if (not refresh):
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map = device)
        tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH, device = device)
    else:
        bnb_config = None if 'cuda' not in device else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16) 
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, device = device)

        model.save_pretrained(LLM_MODEL_PATH, from_pt=True)
        tokenizer.save_pretrained(LLM_TOKENIZER_PATH, from_pt = True)
                

    models_path = '/home/gridsan/afogelson/osfm/saved_results/ai_models/models_urop.csv'
    citations_path = '/home/gridsan/afogelson/osfm/saved_results/citations/citations_0530/results/merged_results.csv'

    selector = LLMModelSelector(model = model, tokenizer = tokenizer, device = device, models_path = models_path)
    citations_df = (pd.read_csv(citations_path)    
                        .sort_values(by=['paperId', 'modelKey', 'classification_order'], ascending=[True, True, False])
                        .drop_duplicates(subset=['paperId', 'modelKey'], keep='first')
                        )
    
    print(f"After dropping duplicates, we found {len(citations_df)}")
    
    citations_df = citations_df[citations_df['classification'].apply(lambda s: s in ('uses', 'extends'))]
    
    print(f"Filtering by only uses and extends, we have {len(citations_df)}")
    
    data = list(zip(citations_df['sentence'], citations_df['modelId'], citations_df['paperId']))
    
    clustered_data = cluster.clusterList(data)
    output_path = f'/home/gridsan/afogelson/osfm/paper_analysis_toolkit/results/modelKeySelection/results_{right_now}_{args.index}_of_{args.workers}.csv'
    f = open(output_path, 'w')
    dict_writer = DictWriter(f, fieldnames=['sentence', 'modelId', 'paperId', 'selectedKey']) 
    dict_writer.writeheader()
    
    for sentence, modelId, paperId in tqdm(clustered_data):
        selected_key = selector.find_model_key(sentence, modelId)
        dict_writer.writerow({'sentence': sentence, 'modelId': modelId, 'paperId': paperId, 'selectedKey': selected_key})
        f.flush()
        
    f.close()

if __name__ == '__main__':
    main()
