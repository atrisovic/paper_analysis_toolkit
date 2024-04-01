from os import walk, path
import regex as re
from nltk.tokenize import sent_tokenize
from typing import List
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def whitespaceless_compare(s1, s2):
    return re.sub(r'[\s\n]*','', s1) == re.sub(r'[\s\n]*','', s2)


class Corpus:
    def __init__(self, directory, extensions):
        self.documents = []
        for root, dirs, files in walk(directory):
            self.documents += [Document(path.join(root, file)) 
                                    for file in files 
                                        if (file.split('.')[-1] in extensions)]

    
class Document:
    def __init__(self, path: str):
        with open(path, "r") as file:
            content = file.read()
            
        self.path = path
        self.text = content
        self.sentences = sent_tokenize(self.text)
                        
                
    '''Filters all senetences by a particular ref'''
    def get_citation_sentences(self, ref: str) -> List[str]:
        return [] if ref is None else [sentence for sentence in self.sentences 
                                       if ref in sentence and sentence[0] != '*']
    
    
    ''' Retrieves appropriate citation based on paper title'''
    def get_citation_ref_from_title(self, title: str):
        numerical_refs = re.findall(f"\* (\[\d+\]).*{title}", self.text)
        string_refs = re.findall(f"\* ([^\\n\)]+[\)]).*{title}", self.text)
        
        if not (len(numerical_refs) <= 1 and len(string_refs) <= 1):
            print(f"{self.path} seems to contain the following title twice: {title}")
        
        ref_number = None if not numerical_refs else numerical_refs[0]
        ref_str = None if not string_refs else re.sub(r'\s*[\(\[]?(\d+)[\)\]]?', r', \1', string_refs[0])
                
        return ref_number or ref_str
    
    
class CitationClassifier:
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    
    def classify(self, sentence):
        return self.classifier(sentence)


if __name__ == '__main__':    
    corpus = Corpus('./Markdown/', ['mmd'])
    results = {}
    for document in tqdm(corpus.documents):
        ref = document.get_citation_ref_from_title(title="Llama: Open and efficient foundation language models")
        results[document.path] = document.get_citation_sentences(ref)

    
    print(len(results))
    
         
