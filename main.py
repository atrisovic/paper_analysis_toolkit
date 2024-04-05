from CitationClassifier import CitationClassifier
from ClassificationCounter import ClassificationCounter
from Corpus import Corpus
import json, pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore") #just for clarity, temporarily

if __name__ == '__main__':

    markdown_file_path = './Markdown/'
    foundation_models_path = 'foundation_models.json'
    logfile = f'logs/logfile_{datetime.now()}.txt'

    classifier = CitationClassifier('allenai/multicite-multilabel-scibert')
    corpus = Corpus(markdown_file_path, extensions = ['mmd'], limit = None)

    with open(foundation_models_path, 'r') as f:
        foundational_models_json = json.load(f)
        keys, titles = list(zip(*[(key, data['title'].replace('\\infty', '∞')) for key, data in foundational_models_json.items()]))

    corpus.findAllPaperRefsAllTitles(titles = titles, keys = keys, classifier = classifier, logfile = logfile)

    with open('pickle/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)