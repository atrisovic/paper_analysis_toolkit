from TextualReference import TextualReference
from CitationClassifier import CitationClassifier
from typing import List
from utils import implies
import regex as re


class Reference:
    def __init__(self, title, key, paper_path, citation = None):
        self.title: str = title.lower()
        self.key: str = key.lower()
        self.paper_path: str = paper_path
        
        self.citations: List[str] = citation
        self.textualReferences: List[TextualReference] = None
        
        self.missing_citation: bool = None
        self.duplicative_citation: bool = None
        self.reference_exists: bool = None
        self.missing_page_fail: bool = None
        
    def __repr__(self):
        return f"Title: {self.title}, Citation: '{self.citations}', missing_citation: {self.missing_citation}"
        
    def getCitationFromContent(self, content: str) -> str:    
        numerical_refs = re.findall(r"\* ([\(\[]\d{1,3}[\)\]]).*" + self.title, content) #  * [38] SOME TEXT HERE title
        string_refs = re.findall(r"\*[\s]+([^\n\)\]]{1,35}[\)\]]+).*" + self.title, content) #  * Name, Extra, (Year) SOME TEXT HERE title
        
        #assert(implies(self.title in content, numerical_refs or string_refs)), f"Found '{self.title}' in {self.paper_path}, but can't link citation."
        self.reference_exists = self.title in content
        self.missing_citation = not implies(self.reference_exists, bool(numerical_refs or string_refs) )
        self.duplicative_citation = len(numerical_refs) > 1 or len(string_refs) > 1

        ref_number = None if not numerical_refs else numerical_refs
        
        if not string_refs:
            ref_str = None
        else:
            ref_str = string_refs + list(map(lambda s: re.sub(r'\s*[\(\[]?(\d{4}[a-z]?)[\)\]]?', r', \1', s), string_refs))
        
        assert(ref_str is None or len(list(filter(lambda s: len(s) >= 100, ref_str))) == 0 ), f"{self.paper_path}, {ref_str}"
        
        self.citations = ref_number or ref_str
        
        return self.citations
    
    def checkMissingPageFailure(self, content: str):
        self.missing_page_fail: bool = content.find('missing_page_fail') >= 0
    
    
    def checkCitationInSentence(self, sentence: str) -> List[str]:
        for citation in self.citations:
            if citation in sentence:
                return True
            
        return False
    
    def getSentencesFromContent(self, all_sentences: List[str]) -> List[str]:
        if self.citations is None:
            self.textualReferences = []
        else:
            self.textualReferences = [TextualReference(sentence) for sentence in all_sentences if (self.checkCitationInSentence(sentence))]
            
        return self.textualReferences

    def classifyAllSentences(self, classifier: CitationClassifier):
        for textualReference in self.textualReferences:
            textualReference.classify(classifier=classifier)
            
    def getAllTextualReferences(self, as_dict = False):
        if (as_dict):
            return [textRef.as_dict() | {'FM_key': self.key, 'FM_title': self.title} for textRef in self.textualReferences]
        else:
            return self.textualReferences