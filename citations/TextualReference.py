from .CitationClassifier import CitationClassifier
from typing import Set

class TextualReference:
    def __init__(self, sentence: str, labels: Set[str]):
        self.sentence: str = sentence
        self.labels: Set[str] = labels
        
        self.classification: str = None
        self.score: float = None

        
    def classify(self, classifier: CitationClassifier):
        model_classification = classifier.classify_text(self.sentence)
        
        is_uses = model_classification in ('uses','extends')
        background_label = 'background' in self.labels

        if (is_uses and background_label):
            self.classification = 'backgroundish'
        else:
            self.classification = model_classification
            
    
    def as_dict(self):
        #TODO this needs to be ordered to include "backgroundish"
        order_class_values = ['backgroundish', 'extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']
        classification_rankings = {val: idx for idx, val in enumerate(order_class_values)}
        return {
                'sentence': self.sentence,
                'classification': self.classification,
                'score': self.score,
                'classification_order': classification_rankings[self.classification]
                }
        