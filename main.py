from CitationClassifier import CitationClassifier
from Corpus import Corpus
from ClassificationCounter import ClassificationCounter
import json, pickle, regex as re


if __name__ == '__main__':
    classifier = None #CitationClassifier('allenai/multicite-multilabel-scibert')
    corpus = Corpus('./Markdown', extensions = ['mmd'], limit = 1000)
    
    with open('foundation_models.json', 'r') as f:
        foundational_models_json = json.load(f)
            # one of the paper titles has a backslash in it. not a permanent solution!
        keys, titles = zip(*[(key, re.escape(data['title'])) for key, data in foundational_models_json.items()])

    corpus.findAllPaperRefsAllTitles(titles = titles, keys = keys, classifier = classifier)
    df = ClassificationCounter(classifier = classifier).getClassificationCounts(corpus = corpus)
        
    

    with open('pickle/df.pkl', 'wb') as f:
        pickle.dump(df, f)
        
    with open('pickle/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
        
    with open('pickle/foundationalModelUseCaseCounts.pkl', 'wb') as f:
        pickle.dump(df, f)