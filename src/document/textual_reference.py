from src.classifiers.CitationClassifier import CitationClassifier
from typing import Set

class TextualReference:
    def __init__(self, sentence: str, labels: Set[str]):
        self.sentence: str = sentence
        self.labels: Set[str] = labels
        
        self.classification: str = None
        self.classification_order: int = None

        
    def classify(self, classifier: CitationClassifier):
        self.classification = classifier.classify_text(self.sentence, self.labels)
        self.classification_order = classifier.getClassificationRanking(self.classification)
            
    
    def as_dict(self):
        return {
                'sentence': self.sentence,
                'classification': self.classification,
                'classification_order': self.classification_order,
                'labels': self.labels
                }

        