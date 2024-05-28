from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import backends, cuda, bfloat16

from src.process.Corpus import Corpus
from src.process.Cluster import Cluster
from datetime import datetime 
from config import MARKDOWN_FILES_PATH, LLM_MODEL_NAME, LLM_MODEL_PATH, LLM_TOKENIZER_PATH
import nltk, logging, argparse

from src.language_models.LLMFullAffiliations import LLMFullAffiliationsPipepline
from src.language_models.LLMInstitutions import LLMInstitutions
from src.prompts.affiliation_prompts import PROMPT3

nltk.download('punkt')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of documents scanned.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('-s', '--seed', default = 0, type = int, help = "Seed used for all random processes. Default is 0.")
    parser.add_argument('--eagerstorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading documents.")

    args = parser.parse_args()
    
    cluster = Cluster(index = args.index, worker_count = args.workers, limit = args.limit, seed = args.seed)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/affiliations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/affiliations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
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
                

        
    # affPipepline = LLMFullAffiliationsPipepline(model  = model, 
    #                                        tokenizer = tokenizer, 
    #                                        device = device, 
    #                                        resultsfile = resultsfile,
    #                                        prompt = PROMPT1,
    #                                        strict = False)
    
    affPipepline = LLMInstitutions(model  = model, 
                                    tokenizer = tokenizer, 
                                    device = device, 
                                    resultsfile = resultsfile,
                                    prompt = PROMPT3,
                                    debug = args.debug)
        
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster = cluster,
                        filter_path = args.filter_file,
                        lazy = not args.eagerstorage,
                        confirm_paper_ref_sections=False)
    
    corpus.getAllAffiliations(pipeline = affPipepline)



if __name__ == '__main__':
    main()
