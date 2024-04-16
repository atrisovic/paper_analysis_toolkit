from .CitationClassifier import CitationClassifier


class TextualReference:
    def __init__(self, sentence):
        self.sentence: str = sentence
        self.classification: str = None
        self.score: float = None
        
    def classify(self, classifier: CitationClassifier):
        self.classification = classifier.classify_text(self.sentence)
    
    def as_dict(self):
        order_class_values = ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']
        classification_rankings = {val: idx for idx, val in enumerate(order_class_values)}
        return {
                'sentence': self.sentence,
                'classification': self.classification,
                'score': self.score,
                'classification_order': classification_rankings[self.classification]
                }
        