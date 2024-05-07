from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import cuda, backends
from typing import Tuple, List, Set, Literal
import regex as re
from .FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from .Parser import OutputParser

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
      
class MistralEnhancedMulticiteClassifier(CitationClassifier):
    def __init__(self, model_checkpoint, llm_model, llm_tokenizer, device = None):    
        device = device or ('mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu')
        self.mistral_pipeline = MistralCitationPipeline(model = llm_model, tokenizer=llm_tokenizer, device = device)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.multicite_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        results = self.multicite_classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        best_label = max(as_dictionary, key=as_dictionary.get)
        
        if (best_label in ('uses', 'extends')):
            llm_label: Classification = self.mistral_pipeline.generateAsModel(input = text)
            if (llm_label is not None):
                return llm_label.classification
            else:
                pass
        
        return best_label
    
    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']
    
    
class Classification(PydanticModel):
    classification: Literal['uses', 'extends', 'background', 'motivation', 'future_work', 'differences']
    
class MistralCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device = None):
        super().__init__(model = model, tokenizer=tokenizer, device=device)
    
        self.outputParser = OutputParser(outputClass = Classification)
    def wrapInstructions(self, input: str):
        return f"""The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. Please response in JSON format 
{{'classification': 'uses | extends | background | motivation | future_work | differences'}}, classifying the following sentence {input}"""   

    def generateAsModel(self, input: str, tolerance= 5, paperId: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            output_object = self.outputParser.parse(results)
             
        
        return output_object