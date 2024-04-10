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
                      {"first": "firstname", "last": "lastname", "gender": "male | female"}...
                  ],
                  "institutions": ["Institution1", "Institution2"],
                  "countries": ["country1", "country2"],
                  "country_explanation": "Explain how you know what country it is from."
                }
                """

    def stripJSONResult(self, response_str: str):
        starting_index = response_str.find('[/INST]') + len('[/INST]')
        end_index = response_str.find('</s>')
        return response_str[starting_index: end_index].strip()


    def textToPrompt(self, text: str) -> TensorType:
        prompt = f'''The following Markdown text has been stripped from the first page of an academic paper. \
                              Please read this text, and return the following information in the JSON format provided: {self.getOutputFormat()}\
                              Do not guess the country or institution. Ensure that it is clearly listed in some identifiable way.
                              The text is as follows: {text}'''

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        encoded_prompt = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        return encoded_prompt


    def classifyFromText(self, text: str) -> dict:
        #format input
        input = self.textToPrompt(text).to(self.device)
        
        #generate response
        generated_ids = self.model.generate(input,
                                              max_new_tokens=1000,
                                              do_sample=True,
                                              pad_token_id = self.tokenizer.eos_token_id)
        raw_output: str = self.tokenizer.batch_decode(generated_ids)[0]

        #strip as json
        json_string_extracted: str = self.stripJSONResult(raw_output)
        
        try:
            json_result = json.loads(json_string_extracted)
        except Exception as e:
            print(f"Error decoding JSON, returning the string itself: {e}")
            json_result = json_string_extracted

        return json_result
