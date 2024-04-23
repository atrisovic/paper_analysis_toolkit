from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import cuda, backends
from typing import Tuple, List

class CitationClassifier:
    def __init__(self):
        pass
    
    def classify_text(self, text: str, *args) -> str:
        pass
    
    def getClassificationOrdering(self) -> List[str]:
        pass
    
    def getClassificationRanking(self, result: str):
        return self.getClassificationOrdering().index(result)
    


class MultiCiteClassifier(CitationClassifier):
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
        device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        results = self.classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        return max(as_dictionary, key=as_dictionary.get)
    
    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']

        

class MultiCiteExtendedClassifider(CitationClassifier):
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
        device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
        
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
    
    def classify_text(self, text, *args) -> str:
        results = self.classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        classification = max(as_dictionary, key=as_dictionary.get)
        
        is_uses = classification in ('uses','extends')
        
        background_label = 'background' in self.labels

        if (is_uses and background_label):
            self.classification = 'usetobackground'
        else:
            self.classification = classification
            
    
    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background', 'usetobackground']
    

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