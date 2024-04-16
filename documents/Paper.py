from citations.Reference import Reference
import regex as re
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import logging
from affiliations.AffiliationClassifier import AffiliationClassifier
from datetime import datetime

logger = logging.getLogger(__name__)

class Paper:
    def __init__(self, path: str, lazy = False):
        self.path: str = path
        self.lazy: bool = lazy
        self.content, self.nonref_section, self.ref_section, self.sentences = None, None, None, None
        
        if (not self.lazy):
            self.getAdjustedFileContent()
            nonref_section, _ = self.getReferenceSectionSplit()
            self.getSentences(nonref_section)
            
        self.setPaperTitle()
        self.setPreAbstract()
        
        assert(self.exactlyOneReferenceSection()), f"Not exactly one reference check. Failing."
                        
        self.references: Dict[str, Reference] = {}
        
        self.name_and_affiliation: dict = None
        
        
    def getAdjustedFileContent(self):
        if (self.content is not None):
            return self.content
    
        with open(self.path, "r", encoding = 'utf-8') as f:
            file_content = ( f.read()
                                .lower()
                                .replace('et al.', 'et al')   #sentence tokenizer mistakes the period for end of sentence
                                )
        normalized = self.normalizeNumericalCitations(file_content)
        
        if (not self.lazy):
            self.content = normalized
        
        return normalized
    
    def setPaperTitle(self, content = None):      
        content = content or self.getAdjustedFileContent()   
        first_line = content.split('\n')[0]
        paper_title = re.sub(r'#','', first_line)
        self.title = paper_title   
        
    def setPreAbstract(self, content = None):
        content = content or self.getAdjustedFileContent()
        match = re.search('#+\s?abstract', content)
        self.pre_abstract = None if match is None else content[:match.start()]
        
    def normalizeNumericalCitations(self, content: str) -> str:
        # [1,2,3,4,5] =====> [1],[2],[3],[4],[5]
        numerical_sequence_citations = [match[0] for match in re.findall('\[((\d+\s*[,;]\s*)+\d+)\]', content)]
        
        for sequence in numerical_sequence_citations:
            corrected_citations = ','.join(f'[{num}]' for num in map(int, re.split('[,;]\s*', sequence)))
            content = re.sub('\[' + sequence + '\]', corrected_citations, content)
            
            
        # [1-5] =====> [1],[2],[3],[4],[5]
        numerical_range_citations = re.findall('\[(\d+-\d+)\]', content)
        for citation in numerical_range_citations:
            n1, n2 = map(int, re.findall('(\d+)-(\d+)', citation)[0])
            if (n2 - n1 > 1000): #arbitrary thresholdß
                continue
            corrected_citations = ','.join(f'[{num}]' for num in range(n1, n2+1))
            content = re.sub('\[' + citation + '\]', corrected_citations, content)
        
        return content
    
    def exactlyOneReferenceSection(self):
        content = self.getAdjustedFileContent()
        ref_section_matches = re.findall('(#+\s?references[\s\S]*?)(?=#+\s*appendix|\Z)', content)
        return len(ref_section_matches) == 1
                    
    def getReferenceSectionSplit(self, content = None):
        if (self.ref_section is not None and self.nonref_section is not None):
            return self.nonref_section, self.ref_section
        
        content = content or self.getAdjustedFileContent()
        
        ref_section_matches = re.findall('(#+\s?references[\s\S]*?)(?=#+\s*appendix|\Z)', content)
        assert(len(ref_section_matches) == 1), f"Length of reference section matches object is {len(ref_section_matches)}, should be 1."

        
        reference_section = ref_section_matches[0]
        nonref_section = content.replace(reference_section, '')
    
        if (not self.lazy):
            self.nonref_section, self.ref_section = nonref_section, reference_section
        
        return nonref_section, reference_section 
    
    def getSentences(self, nonref_section):
        if (self.sentences is not None):
            return self.sentences
        
        all_sentences = sent_tokenize(nonref_section)
        
        if (not self.lazy):
            self.sentences = all_sentences
        
        return all_sentences
        
    def getReferenceFromTitle(self, title, key, classifier = None) -> Reference:
        if (self.content is None):
            logger.debug(f'No content found for page {self.path}, pulling the document again.')
        
        content = self.getAdjustedFileContent()
        nonref_section, ref_section = self.getReferenceSectionSplit()
        all_sentences = self.getSentences(nonref_section)
        reference = Reference(title = title, key = key, paper_path = self.path)
        reference.checkMissingPageFailure(content = content)
        reference.getCitationFromContent(content = ref_section)
        reference.getSentencesFromContent(all_sentences=all_sentences)
        
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
        
    
    def findNamesAndAffiliations(self, classifier: AffiliationClassifier) -> dict:
        self.name_and_affiliation = classifier.classifyFromTextEnsureJSON(self.pre_abstract)
        return self.name_and_affiliation