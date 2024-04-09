from Reference import Reference
import regex as re
from typing import List, Dict
from nltk.tokenize import sent_tokenize



class Paper:
    def __init__(self, path: str):
        self.path: str = path
        self.content: str = self.retrieveAllContent()
        
        nonref_section, ref_section = self.retrieveAndFormatContent()
        self.nonref_section = nonref_section
        self.ref_section = ref_section
        
        
        first_line = self.nonref_section.split('\n')[0]
        self.title: str = re.sub(r'#','', first_line)
        
        self.all_sentences: List[str] = sent_tokenize(self.nonref_section)
                
        self.references: Dict[str, Reference] = {}
    
    def fix_numerical_sequence_citations(self, content: str):
        numerical_sequence_citations = [match[0] for match in re.findall('\[((\d+\s*[,;]\s*)+\d+)\]', content)]
        for sequence in numerical_sequence_citations:
            corrected_citations = ','.join(f'[{num}]' for num in map(int, re.split('[,;]\s*', sequence)))
            content = re.sub('\[' + sequence + '\]', corrected_citations, content)
            
        numerical_range_citations = re.findall('\[(\d+-\d+)\]', content)
        for citation in numerical_range_citations:
            n1, n2 = re.findall('(\d+)-(\d+)', citation)[0]
            corrected_citations = ','.join(f'[{num}]' for num in range(int(n1), int(n2)+1))
            content = re.sub('\[' + citation + '\]', corrected_citations, content)
        
        return content
        
    def retrieveAllContent(self):
        with open(self.path, "r") as f:
            file_content = f.read()
        remove_et_all = file_content.lower().replace('et al.', 'et al')
        numerical_sequence_citations_fixed = self.fix_numerical_sequence_citations(remove_et_all)
        
        return numerical_sequence_citations_fixed
        
            
    def retrieveAndFormatContent(self):
        reference_section = re.findall('(#+\s?references[\s\S]*?)(?=#+\s*appendix|\Z)', self.content)
        
        assert(len(reference_section) == 1), reference_section
        reference_section = reference_section[0]
        
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