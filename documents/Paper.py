from citations.Reference import Reference
import regex as re
from typing import List, Dict, Optional
from nltk.tokenize import sent_tokenize
import logging
from affiliations.AffiliationClassifier import AffiliationClassifier
from datetime import datetime
from utils.functional import implies
import numpy as np
from os.path import basename


logger = logging.getLogger(__name__)

class ReferenceSectionCountException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class Paper:
    def __init__(self, path: str, lazy = False, confirm_reference_section = True):
        self.path: str = path
        self.lazy: bool = lazy

        self.sections: List[str] = None
        self.sentences: Dict[str, List[str]] = None
        self.references: Dict[str, Reference]= {}
        self.name_and_affiliation: dict = None
        
    
        if (not self.lazy):
            self.getSections()
            self.getSentences()
            
        self.setPaperTitle()
        self.setPreAbstract()
                                
        if (confirm_reference_section):
            self.checkReferenceSectionCount()
        
        
    def getContent(self):
        sections = self.getSections()
        content =  ' '.join(section['content'] for section in sections)
        
        #assert(len(content) > 0), f"Found no content within sections."
        return content
    
    def getSections(self, heading_prefix = '##') -> List[dict]:
        if (self.sections is not None):
            return self.sections
        
        with open(self.path, "r", encoding = 'utf-8') as f:
            file_content = ( f.read()
                                .lower()
                                .replace('et al.', 'et al')   #sentence tokenizer mistakes the period for end of sentence
                                )
        content = self.normalizeNumericalCitations(file_content)
        
        sections = [{'content': f'{heading_prefix} ' + section} for section in filter(None, re.split(f"\n{heading_prefix}(?=[^#])", content))]
        
        for section in sections:
            section['labels'] = self.labelSection(section['content'])
            
        if sections:
            sections[0]['labels'].add('first')
        else:
            logger.debug(f"No sections found for file at {self.path}. Content size is {len(content)}.")
        
        if (not self.lazy):
            self.sections = sections
            
        return sections
    
    
    def getSentences(self):
        if (self.sentences is not None):
            return self.sentences
        
        nonref_sections = self.getSectionsByLabel(label = 'reference', complement=True, join = False)      
        
        all_sentences = {}
        for section in nonref_sections:  
            sentences = sent_tokenize(section['content'])
            all_sentences |= {sentence: section['labels'] for sentence in sentences}
        
        if (not self.lazy):
            self.sentences = all_sentences
        
        return all_sentences
    
    def getSectionsByLabel(self, label: str, complement: bool = False, join: bool = True):
        sections = self.getSections()
        
        selection_function = lambda labels: (not complement) == (label in labels)
        
        valid_sections = [section for section in sections if selection_function(section['labels'])]
        
        if join:
            return ''.join([section['content'] for section in valid_sections])
        
        return valid_sections
    
    def setPaperTitle(self, content = None):      
        content = content or self.getContent()   
        first_line = content.split('\n')[0]
        paper_title = re.sub(r'#','', first_line)
        self.title = paper_title   
        
    def setPreAbstract(self, content = None):
        content = content or self.getContent()
        match = re.search(r'#+\s?abstract', content)
        self.pre_abstract = None if match is None else content[:match.start()]
        
    def normalizeNumericalCitations(self, content: str) -> str:
        # [1,2,3,4,5] =====> [1],[2],[3],[4],[5]
        numerical_sequence_citations = [match[0] for match in re.findall(r'\[((\d+\s*[,;]\s*)+\d+)\]', content)]
        
        for sequence in numerical_sequence_citations:
            corrected_citations = ','.join(f'[{num}]' for num in map(int, re.split(r'[,;]\s*', sequence)))
            content = re.sub(r'\[' + sequence + r'\]', corrected_citations, content)
            
            
        # [1-5] =====> [1],[2],[3],[4],[5]
        numerical_range_citations = re.findall(r'\[(\d+-\d+)\]', content)
        for citation in numerical_range_citations:
            n1, n2 = map(int, re.findall(r'(\d+)-(\d+)', citation)[0])
            if (n2 - n1 > 1000): #arbitrary thresholdß
                continue
            corrected_citations = ','.join(f'[{num}]' for num in range(n1, n2+1))
            content = re.sub(r'\[' + citation + r'\]', corrected_citations, content)
        
        return content
    
    def getReferenceFromTitle(self, title, key, classifier = None) -> Reference:        
        content = self.getContent()
        
        if (content is None):
            logger.debug(f'No content found for page {self.path}, pulling the document again.')

        reference = Reference(title = title, key = key, paper_path = self.path)
        reference.checkMissingPageFailure(content = content)
        
        reference.getCitationFromContent(content = self.getSectionsByLabel(label = 'reference'))
        reference.getTextualReferencesFromSentences(all_sentences=self.getSentences())
        
        if classifier:
            reference.classifyAllSentences(classifier = classifier)

        self.references[key] = reference
        
        return reference
    
    def getAllTextualReferences(self, as_dict = False) -> List[dict] | List[Reference]:
        if (as_dict):
            return [text_ref | {'paperId': basename(self.path)} for title, reference in self.references.items() for text_ref in reference.getAllTextualReferences(as_dict = True)]
        else:
            return [text_ref for title, reference in self.references.items() for text_ref in reference.getAllTextualReferences()]
    
    def getNamesAndAffiliations(self, classifier: AffiliationClassifier) -> dict:
        self.name_and_affiliation = classifier.classifyFromTextEnsureJSON(self.pre_abstract)
        return self.name_and_affiliation
    
    def getGenericHeadingCheckerFunction(self, *args):    
        generate_regex = lambda s: r'(#+\s*(?:\d*|[ivx]*)\.?\s*' + f"{s})"
        check_all_regex = lambda s: np.array([bool(re.findall(generate_regex(header),s)) 
                                                for header in args]
                                             ).any()
        return check_all_regex
        
    def labelSection(self, content: str):    
        mappings = {
                            'reference': self.getGenericHeadingCheckerFunction('references', 'citations','bibliography'),
                            'method': self.getGenericHeadingCheckerFunction('methodology', 'method', 'approach', 'experiment'),
                            'abstract': self.getGenericHeadingCheckerFunction('abstract'),
                            'appendix': self.getGenericHeadingCheckerFunction('appendix'),
                            'background': self.getGenericHeadingCheckerFunction('related work', 'background'),
                            'conclusion': self.getGenericHeadingCheckerFunction('conclusion', 'discussion'),
                            'introduction': self.getGenericHeadingCheckerFunction('introduction'),
                            'result': self.getGenericHeadingCheckerFunction('result')
        }
        
        labels = {label for label, func in mappings.items() if func(content)}
        return labels
    
    def checkReferenceSectionCount(self):
        ref_sections = self.getSectionsByLabel(label = 'reference', join = False)
        if (len(ref_sections) != 1):
            raise ReferenceSectionCountException(message = f"Found {len(ref_sections)} sections labeled 'reference'.") 