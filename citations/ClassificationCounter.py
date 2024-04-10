from .CitationClassifier import CitationClassifier
from .Corpus import Corpus
import pandas as pd

class ClassificationCounter:
    def __init__(self, classifier: CitationClassifier):
        self.classifier = classifier
        
    def getClassificationCounts(self, corpus: Corpus):
        df = pd.DataFrame.from_dict(corpus.getAllTextualReferences(as_dict = True))
        df['classification_ranking'] = df.groupby(['FM_key', 'paper'])['classification_order'].rank(method='min')
        classification_counts = (df[df['classification_ranking'] == 1]
                                                .groupby(['FM_key', 'classification'])['paper']
                                                .nunique()
                                                .reset_index()
                                                .rename(columns={'paper':'count'})
                                                .pivot(index='FM_key', columns='classification', values='count')
                                                .fillna(0)
                                            )
        return classification_counts
