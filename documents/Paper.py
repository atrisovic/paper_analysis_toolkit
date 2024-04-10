from citations.Reference import Reference
import regex as re
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import logging
from affiliations.AffiliationClassifier import AffiliationClassifier


logger = logging.getLogger(__name__)

class Paper:
    def __init__(self, path: str):
        self.path: str = path
        self.content: str = self.adjustFileContent()
        
        self.nonref_section, self.ref_section = self.splitByReferenceSection()
        
        self.title: str = self.getPaperTitle()
        
        self.all_sentences: List[str] = sent_tokenize(self.nonref_section)
                
        self.references: Dict[str, Reference] = {}
        
        self.name_and_affiliation: dict = None
        
    def adjustFileContent(self):
        with open(self.path, "r") as f:
            file_content = ( f.read()
                                .lower()
                                .replace('et al.', 'et al')   #sentence tokenizer mistakes the period for end of sentence
                                )
        normalized = self.normalizeNumericalCitations(file_content)
        
        return normalized
    
    def getPaperTitle(self):
        first_line = self.nonref_section.split('\n')[0]
        return re.sub(r'#','', first_line)

    def getPreAbstract(self):
        match = re.search('#+\s?Abstract', self.content)
        return None if match is None else self.content[:match.start()]
    
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
            corrected_citations = ','.join(f'[{num}]' for num in range(n1, n2+1))
            content = re.sub('\[' + citation + '\]', corrected_citations, content)
        
        return content
                    
    def splitByReferenceSection(self):
        ref_section_matches = re.findall('(#+\s?references[\s\S]*?)(?=#+\s*appendix|\Z)', self.content)
        assert(len(ref_section_matches) == 1), f"Length of reference section matches object is {len(ref_section_matches)}, should be 1."
        
        reference_section = ref_section_matches[0]
        nonref_section = self.content.replace(reference_section, '')
        
        return nonref_section, reference_section 
        
        
    def getReferenceFromTitle(self, title, key, classifier = None) -> Reference:
        reference = Reference(title = title, key = key, paper_path = self.path)
        reference.checkMissingPageFailure(content = self.content)
        reference.getCitationFromContent(content = self.ref_section)
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
        
    
    def findNamesAndAffiliations(self, classifier: AffiliationClassifier)->dict:
        pre_abstract = self.getPreAbstract()
        self.name_and_affiliation = classifier.classifyFromText(pre_abstract)
        return self.name_and_affiliation