from typing import List

class CitationClassifier:
    def __init__(self):
        pass
    
    def classify_text(self, text: str, *args) -> str:
        pass
    
    def getClassificationOrdering(self) -> List[str]:
        pass
    
    def getClassificationRanking(self, result: str):
        return self.getClassificationOrdering().index(result)
    
      

