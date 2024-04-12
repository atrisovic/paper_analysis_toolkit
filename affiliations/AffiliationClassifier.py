from torch import TensorType
import regex as re
import json
from os.path import join
from typing import List
from tqdm import tqdm

class AffiliationClassifier:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device



    def getOutputFormat(self):
      return """
{
    "contributors": [
        {
            "first": "firstname", 
            "last": "lastname", 
            "gender": "male | female"
        }...
    ],
    "institutions": ["Institution1", "Institution2"],
    "countries": ["country1", "country2"]
}
                """
                
    def getExamples(self):
        example_text1 = '''# on the helpfulness of large language models

bill ackendorf, Jolene baylor

yuxin shu, khalid saifullah

alex fogelson \({}^{\dagger}\), ana trivosic, neil thompson\({}^{\ddagger}\), bob dilan

\({}^{\ddagger}\) new york university, massachusetts institute of technology'''
        example_response1 = '''
{
    "contributors": [
        {"first": "Bill", "last": "Ackendorf", "gender": "male"},
        {"first": "Jolene", "last": "Baylor", "gender": "female"},
        {"first": "Yuxin", "last": "Shu", "gender": "female"},
        {"first": "Khalid", "last": "Saifullah", "gender": "male"},
        {"first": "Alex", "last": "Fogelson", "gender": "male"},
        {"first": "Ana", "last": "Trivosic", "gender": "female"},
        {"first": "Neil", "last": "Thompson", "gender": "male"}],
    "institutions": ["New York University", "Massachusetts Institute of Technology"],
    "countries": ["United States"]
}'''
        
        return [(example_text1, example_response1)]
                
    def stripJSONResult(self, response_str: str):
        starting_index = response_str.rfind('[/INST]') + len('[/INST]')
        end_index = response_str.rfind('</s>')
        return response_str[starting_index: end_index].strip()
    
    def formatPrompt(self, text: str):
        return f'''Please read this text, and return the following information in the JSON format provided: {self.getOutputFormat()}\
                    The output should match exactly the JSON format given. The text is as follows: {text}'''

    def textToPrompt(self, text: str) -> TensorType:

        messages = []
        for example_text, example_response in self.getExamples():
            messages.append({"role": "user", "content": self.formatPrompt(example_text)})
            messages.append({"role": "assistant", "content": example_response})

        messages.append(
            {"role": "user", "content": self.formatPrompt(text)}
        )
        
        encoded_prompt = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        return encoded_prompt


    def classifyFromText(self, text: str) -> dict:
        #format input
        input = self.textToPrompt(text).to(self.device)
        
        #generate response
        generated_ids = self.model.generate(input,
                                              max_new_tokens=1000,
                                              do_sample=True,
                                              pad_token_id = self.tokenizer.eos_token_id,
                                              temperature = .25)
        raw_output: str = self.tokenizer.batch_decode(generated_ids)[0]
        #strip as json
        json_string_extracted: str = self.stripJSONResult(raw_output)

        return json_string_extracted
    
    
    def classifyFromTextEnsureJSON(self, text: str, tolerance = 5) -> dict:
        counter = 0
        json_result = None
        while (counter < tolerance and not json_result):
            counter += 1
            json_string_results = self.classifyFromText(text)
            try:
                json_result = json.loads(json_string_results)
            except:
                pass
        return json_result or self.stripJSONStructure(json_result)
    
    
    def stripJSONStructure(self, json_object: dict) -> dict:
        new_dict = dict()
        new_dict["institutions"] = json_object.get("institutions")
        new_dict["countries"] = json_object.get("countries")
        new_dict["contributors"] = []
        
        for person in json_object.get("contributors"):
            new_person = {}
            for attribute in ['first', 'last', 'gender']:
                new_person[attribute] = person.get(attribute)
        
        return new_dict