from src.classifier.MultiCiteClassifier import MultiCiteClassifier
from src.classifier.MistralEnhancedMulticiteClassifier import MistralEnhancedMulticiteClassifier
from src.analysis.Agglomerator import RankedClassificationCountsYearly, RankedClassificationCounts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.process.FoundationModel import FoundationModel
from src.process.Corpus import Corpus
import pickle
import warnings, logging
from datetime import datetime
import argparse
from config import *
from src.functional import extract_paper_metadata
import pandas as pd
from torch import cuda, backends, bfloat16

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of foundation models analyzed.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('--lazystorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading classes.")

    args = parser.parse_args()
    
    assert(args.limit is None or args.workers <= args.limit)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/citations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/citations/results_{right_now}_worker{args.index}of{args.workers}.csv"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
    
    device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device = {device}")
    
    
    bnb_config = None if device != 'cuda' else BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=bfloat16) 
    
    refresh = False
    try:
        assert(not refresh)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map = device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH, device = device)
    except:
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, device = device)

        model.save_pretrained(LLM_MODEL_PATH, from_pt=True)
        tokenizer.save_pretrained(LLM_TOKENIZER_PATH, from_pt = True)
        
    classifier = MistralEnhancedMulticiteClassifier(model_checkpoint=CITATION_MODEL_PATH,llm_model=model,llm_tokenizer=tokenizer, device=device)
    
    #classifier = MultiCiteClassifier(model_checkpoint=CITATION_MODEL_PATH)
    
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster_info = (args.index, args.workers), 
                        foundation_model_limit = args.limit, 
                        filter_path=args.filter_file, 
                        lazy = args.lazystorage,
                        paper_years=extract_paper_metadata(OPEN_ACCESS_PAPER_XREF)
                        )

    models = FoundationModel.modelsFromJSON(FOUNDATION_MODELS_PATH)
          
    corpus.findAllReferencesAllModels(models = models,
                                     classifier = classifier,
                                     resultsfile = resultsfile)

        
if __name__ == '__main__':
    main()