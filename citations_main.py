from citations.CitationClassifier import MultiCiteClassifier
from citations.Agglomerator import RankedClassificationCountsYearly
from citations.FoundationModel import FoundationModel
from documents.Corpus import Corpus
import json, pickle
import warnings, logging
from datetime import datetime
import argparse
from config import FOUNDATION_MODELS_PATH, MARKDOWN_FILES_PATH, CITATION_MODEL_PATH, OPEN_ACCESS_PAPER_XREF
from utils.functional import extract_paper_metadata

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of foundation models analyzed.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('--lazystorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading documents.")

    args = parser.parse_args()
    
    assert(args.limit is None or args.workers <= args.limit)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/citations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/citations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
    classifier = MultiCiteClassifier(CITATION_MODEL_PATH)
    
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster_info = (args.index, args.workers), 
                        foundation_model_limit = args.limit, 
                        filter_path=args.filter_file, 
                        lazy = args.lazystorage,
                        paper_years=extract_paper_metadata(OPEN_ACCESS_PAPER_XREF)
                        )

    models = FoundationModel.modelsFromJSON(FOUNDATION_MODELS_PATH)
      
    corpus.findAllPaperRefsAllTitles(models = models,
                                     classifier = classifier, 
                                     resultsfile = resultsfile,
                                     agglomerator=RankedClassificationCountsYearly())

    with open(f'pickle/corpus{args.index if args.workers > 1 else ""}_{right_now}.pkl', 'wb') as f:
        pickle.dump(corpus, f)
        
if __name__ == '__main__':
    main()