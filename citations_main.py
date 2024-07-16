from src.process.FoundationModel import FoundationModel
from src.process.Corpus import Corpus
from src.process.Cluster import Cluster
from src.prompts.citation_prompts import PROMPT2
from load_classifier import load_classifier

import warnings, logging
from datetime import datetime
import argparse
from config import *
from src.functional import extract_paper_metadata
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of foundation models analyzed.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('-s', '--seed', default = 0, type = int, help = "Seed used for all random processes. Default is 0.")
    parser.add_argument('--lazystorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading classes.")

    args = parser.parse_args()
    cluster = Cluster(index = args.index, worker_count = args.workers, limit = args.limit, seed = args.seed)
    
    assert(args.limit is None or args.workers <= args.limit)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/citations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/citations/results_{right_now}_worker{args.index}of{args.workers}.csv"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
    classifier = None #load_classifier(prompt = PROMPT2)

        
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster = cluster,
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