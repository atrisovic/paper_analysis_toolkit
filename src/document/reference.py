from src.document.textual_reference import TextualReference
from src.functional import implies
from src.process.FoundationModel import FoundationModel

from typing import List, Dict, Set

import regex as re


class Reference:
    def __init__(self, model: FoundationModel, paperId: str, context_size = 3):
        self.model: FoundationModel = model
        self.paperId: str = paperId
        
        self.citations: List[str] = []
        self.textualReferences: List[TextualReference] = None
        
        self.missing_citation: bool = None
        self.duplicative_citation: bool = None
        self.reference_exists: bool = None
        self.missing_page_fail: bool = None
        self.context_size = context_size
        
    def __repr__(self):
        return f"Title: {self.model.title}, Citation: '{self.citations}', missing_citation: {self.missing_citation}"
        
    def getCitationFromContent(self, content: str) -> str:    
        numerical_refs = re.findall(r"\* ([\(\[]\d{1,3}[\)\]]).*" + self.model.title, content) #  * [38] SOME TEXT HERE title
        string_refs = re.findall(r"\*[\s]+([^\n\)\]]{1,35}[\)\]]+).*" + self.model.title, content) #  * Name, Extra, (Year) SOME TEXT HERE title
        
        #assert(implies(self.model.title in content, numerical_refs or string_refs)), f"Found '{self.model.title}' in {self.paperId}, but can't link citation."
        self.reference_exists = self.model.title in content
        self.missing_citation = not implies(self.reference_exists, bool(numerical_refs or string_refs) )
        self.duplicative_citation = len(numerical_refs) > 1 or len(string_refs) > 1

        ref_number = None if not numerical_refs else numerical_refs
        
        if not string_refs:
            ref_str = None
        else:
            ref_str = string_refs + list(map(lambda s: re.sub(r'\s*[\(\[]?(\d{4}[a-z]?)[\)\]]?', r', \1', s), string_refs))
        
        assert(ref_str is None or len(list(filter(lambda s: len(s) >= 100, ref_str))) == 0 ), f"{self.paperId}, {ref_str}"
        
        self.citations = ref_number or ref_str
        
        return self.citations
    
    def checkMissingPageFailure(self, content: str):
        self.missing_page_fail: bool = content.find('missing_page_fail') >= 0
        
    
    def addCitationBrackets(self, sentence: str) -> str:
        for citation in self.citations:
            if citation in sentence:
                return sentence.replace(citation, f'<cite>{citation}</cite>')
        return sentence
    
    def checkCitationInSentence(self, sentence: str) -> List[str]:
        for citation in self.citations:
            if citation in sentence:
                return True
        
        return False
    
    def getTextualReferencesFromSentences(self, all_sentences: Dict[str, Set[str]]) -> List[str]:
        if self.citations is None or len(all_sentences) == 0:
            self.textualReferences = []
            return
        
        safe_index = lambda input_list, index: input_list[index] if index in range(0, len(input_list)) else ''
        window_iterator = lambda input_list, size: ([safe_index(input_list, k) for k in range(index - (size//2), index + (size//2) + (size % 2))] for index in range(len(input_list)))
        
        sentences, labels = zip(*all_sentences.items())
        
        main_idx = self.context_size//2
        self.textualReferences = []
        for sentences, labels in zip(window_iterator(sentences, self.context_size), labels):
            if self.checkCitationInSentence(sentences[main_idx]):
                sentences[main_idx] = self.addCitationBrackets(sentences[main_idx])
                newTextRef = TextualReference(' '.join(sentences), labels=labels) 
                self.textualReferences.append(newTextRef)
        
        return self.textualReferences

    def classifyAllSentences(self, classifier):
        for textualReference in self.textualReferences:
            textualReference.classify(classifier=classifier)
            
    def getAllTextualReferences(self, as_dict = True):
        if (as_dict):
            return [textRef.as_dict() | self.model.as_dict() for textRef in self.textualReferences]
        else:
            return self.textualReferences