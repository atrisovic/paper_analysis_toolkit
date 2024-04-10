from citations.CitationClassifier import CitationClassifier
from citations.Corpus import Corpus
import json, pickle
import warnings, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def main():    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/logfile_{right_now}.log"
    resultsfile = f"results/{right_now}.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)

    markdown_file_path = './data/Markdown/'
    foundation_models_path = './data/foundation_models.json'
    classifier = CitationClassifier('allenai/multicite-multilabel-scibert')
    corpus = Corpus(markdown_file_path, extensions = ['mmd'], limit = None)

    with open(foundation_models_path, 'r') as f:
        foundational_models_json = json.load(f)
        keys, titles = list(zip(*[(key, data['title'].replace('\\infty', '∞')) for key, data in foundational_models_json.items()]))

    corpus.findAllPaperRefsAllTitles(titles = titles, keys = keys, classifier = classifier, resultsfile = resultsfile)

    with open('pickle/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
        
        
        
if __name__ == '__main__':
    main()