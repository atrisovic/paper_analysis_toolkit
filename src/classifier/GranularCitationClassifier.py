from typing import List, Set
import regex as re
from src.classifier.CitationClassifier import CitationClassifier


class GranularCitationClassier(CitationClassifier):
    def __init__(self):
        pass
    
    def classify_text(self, text: str, labels: Set[str]) -> str:
        not_in_methods = labels.intersection({'background', 'result', 'introduction'})
        in_methods = 'method' in labels
        
        if not in_methods:
            return 'none'

        fine_tune = bool(re.findall(r'fine(ly)?[\-\s]?tun', text, re.IGNORECASE))
        n_shot = bool(re.findall(r'(?:few|(?<!\d)\d|\sn|one|two|three|four|five|zero)(?:\-|\s{1})shot', text, re.IGNORECASE))
        prompt = bool(re.findall(r'prompt(ing)?[\-\s]?(?:engineer|tun)', text, re.IGNORECASE))
        cot = bool(re.findall(r'chain[s]?[\-\s]?of[\-\s]?thought', text, re.IGNORECASE) + 
                    re.findall(r'(?:\s+|^)cot(?:[\.\!\?,\-\s]+|$)', text, re.IGNORECASE) +
                    re.findall(r'step[\-\s]?by[\-\s]?step( reason)?', text, re.IGNORECASE)
        )
        rag = bool(re.findall(r'(retrieval[\-\s]?(augment(?:ed)?|enhance(?:d)?|base(?:d)?)[\-\s]?generat)', text, re.IGNORECASE) + 
                re.findall(r'(?:\s+|^)rag(?:[\.\!\?,\-\s]+|$)', text, re.IGNORECASE)
                    )

        
        mapping = {'fine-tune': fine_tune, 
                'n-shot': n_shot, 
                'prompt-eng': prompt,
                'cot': cot,
                'rag': rag}
        
        for classification in self.getClassificationOrdering():
            if mapping.get(classification):
                return classification
        
        return 'none'
        
    
    def getClassificationOrdering(self) -> List[str]:
        return ['fine-tune', 'n-shot', 'prompt-eng', 'cot', 'rag', 'none']
    
    def getClassificationRanking(self, result):
        return  self.getClassificationOrdering().index(result)