from citations.CitationClassifier import CitationClassifier
from documents.Corpus import Corpus
import json, pickle
import warnings, logging
from datetime import datetime
import argparse

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Max amount of documents to process. Default is None.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('--lazystorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading documents.")

    args = parser.parse_args()
    
    assert(args.limit is None or args.workers <= args.limit)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/citations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/citations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)

    markdown_file_path = './data/Markdown/'
    foundation_models_path = './data/foundation_models.json'
    
    
    
    classifier = CitationClassifier('allenai/multicite-multilabel-scibert')
    corpus = Corpus(markdown_file_path, 
                        extensions = ['mmd'], 
                        cluster_info = (args.index, args.workers), 
                        foundation_model_limit = args.limit, 
                        filter_path=args.filter_file, 
                        lazy = args.lazystorage)

    with open(foundation_models_path, 'r') as f:
        foundational_models_json = json.load(f)
        keys, titles = list(zip(*[(key, data['title'].replace('\\infty', '∞')) for key, data in foundational_models_json.items()]))
        keys, titles = list(keys), list(titles)

    corpus.findAllPaperRefsAllTitles(titles = titles, keys = keys, classifier = classifier, resultsfile = resultsfile)

    with open(f'pickle/corpus{args.index if args.workers > 1 else ""}.pkl', 'wb') as f:
        pickle.dump(corpus, f)
        
        
        
if __name__ == '__main__':
    main()