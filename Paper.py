from Reference import Reference
import regex as re
from typing import List, Dict
from nltk.tokenize import sent_tokenize



class Paper:
    def __init__(self, path: str):
        with open(path, "r") as file:
            file_content = file.read()

        self.path: str = path
        
        first_line = file_content.split('\n')[0]
        self.title: str = re.sub(r'#','', first_line)
        
        self.content: str = file_content.lower()
        
        self.all_sentences: List[str] = sent_tokenize(self.content)
        
        self.references: Dict[str, Reference] = {}
        
    def getReferenceFromTitle(self, title, key, classifier = None) -> Reference:
        reference = Reference(title = title, key = key, paper_path = self.path)
        reference.getCitationFromContent(content = self.content)
        reference.getSentencesFromContent(all_sentences=self.all_sentences)
        
        if classifier:
            reference.classifyAllSentences(classifier = classifier)

        self.references[key] = reference
        
        return reference
    
    def getAllReferences(self):
        return [item for _, item in self.references.items()]
    
    def getAllTextualReferences(self, as_dict = False) -> List[dict] | List[Reference]:
        if (as_dict):
            return [text_ref | {'paper': self.title} for title, reference in self.references.items() for text_ref in reference.getAllTextualReferences(as_dict = True)]
        else:
            return [text_ref for title, reference in self.references.items() for text_ref in reference.getAllTextualReferences()]