from typing import List, Set
from src.classifier.CitationClassifier import CitationClassifier


class LocationBasedClassifier(CitationClassifier):
    def __init__(self):
        pass
    
    def getClassificationOrdering(self) -> List[str]:
        return ['method', 'result', 'background', 'introduction']

    def classify_text(self, text, labels, *args) -> str:
        if (not labels.intersection({'background', 'introduction', 'result'})):
            return 'method'
            
        elif ('introduction' in labels):
            return 'introduction'
                    
        elif ('result' in self.labels):
            return 'result'
        
        elif ('background' in self.labels):
            return 'background'
        
        return 'NONE'