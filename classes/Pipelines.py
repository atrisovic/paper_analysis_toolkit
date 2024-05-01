from pydantic import BaseModel as PydanticModel
from classes.Affiliations import PaperAffiliations, Contributor, Institution
from transformers import Pipeline
import logging, json
from typing import Dict, List, Union
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from transformers import Pipeline



logger = logging.getLogger(__name__)


class FewShotExample(PydanticModel):
    input: str
    output: str
    
    
# just want to log here the reason we didn't continue using the fewshot functionality from langchain
# firstly, it had some strange behvaior with curly brackets that was a bit clunky to get around
# secondly, pipelines don't seem to include instruction tuning, which is pretty darn important
class FewShotPipeline:
    pipeline: Pipeline
    examples: List[Dict]
    
    
    def __init__(self, model, tokenizer, device = None, outputClass: PydanticModel = PaperAffiliations):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = []
        self.device = device
        self.outputClass = outputClass
        
    def addExample(self, question: str, answer: Union[str, PydanticModel]):
        if (isinstance(answer, PydanticModel)):
            example = FewShotExample(input = question, output = answer.model_dump_json(indent=1))
        else:
            example = FewShotExample(input = question, output = answer)
            
        self.examples.append(example)   
        
    def getSchema(self):
        return json.dumps(self.outputClass.model_json_schema()['$defs'], indent=1)
    
    def wrapInstructions(self, input: str):
        return input
        return f"""Please read this text, and return the following information in the JSON format provided: 
{self.getSchema()}\n
The output should match exactly the JSON format below. The text is as follows:\n\n {input}"""      
    
    
    def generateZeroShotPrompt(self, input: str, output: str = None):
        zero_shot = [{"role": "user", "content": self.wrapInstructions(input)}]
        
        if output:
            zero_shot.append({"role": "assistant", "content": output})
        
        return zero_shot

    def getFewShotPrompt(self, input: str, max_examples: int = None):
        few_shots = []
        for example in self.examples[:max_examples]:
            few_shots += self.generateZeroShotPrompt(example.input, example.output)
            
        few_shots += self.generateZeroShotPrompt(input = input, output = None)
            
        return few_shots
        
        
    
    def generate(self, input: str, max_examples: int = None):
        few_shot = self.getFewShotPrompt(input)
        encodeds = self.tokenizer.apply_chat_template(few_shot, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=100, pad_token_id = self.tokenizer.eos_token_id, do_sample=True, temperature=.5)
        decoded = self.tokenizer.batch_decode(generated_ids)[0]
    
        return decoded
    
    
    
    
    
    
class AffiliationsPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device, outputClass: PydanticModel = PaperAffiliations, resultsfile: str = None):
        super().__init__(model = model, tokenizer=tokenizer, device = device, outputClass=outputClass)
        
        
        self.resultsfile = resultsfile
        
        for question, answer in self.getExamples():
            pass #self.addExample(question=question, answer=answer)
            

    def getExamples(self):
        example_text = "# on the helpfulness of large language models\n\nbill ackendorf, Jolene baylor\n\nyuxin shu, khalid saifullah\n\nalex fogelson ({}^{\\dagger}), ana trisovic, neil thompson({}^{\\ddagger}), bob dilan\n\n({}^{\\ddagger}) new york university, massachusetts institute of technology"

        paperAffiliations = PaperAffiliations(
                        contributors = [ 
                                        Contributor(first = "Bill", last= "Ackendorf", gender= "male"),
                                        Contributor(first= "Jolene", last= "Baylor", gender= "female"),
                                        Contributor(first= "Yuxin", last= "Shu", gender= "female"),
                                        Contributor(first= "Alex", last= "Fogelson", gender= "male"),
                                        Contributor(first= "Ana", last= "Trisovic", gender= "female"),
                                        Contributor(first= "Neil", last= "Thompson", gender= "male"),
                                ],
                        institutions = [
                                        Institution(name = "New York University", type = "academic"),
                                        Institution(name = "Massachusetts Institute of Technology", type =  "academic")
                                ],
                        countries = ["United States"]
        )
        
        return {example_text: paperAffiliations}.items()
    
    def getSchema(self):
        paperAffiliations = PaperAffiliations(
                        contributors = [ 
                                        Contributor(first = "firstname", last= "lastname", gender= "male"),
                                        Contributor(first= "firstname", last= "lastname", gender= "female")
                                ],
                        institutions = [
                                        Institution(name = "University of City", type = "academic"),
                                        Institution(name = "Major Company", type =  "industry")
                                ],
                        countries = ["United States", "China", "Other Countries"]
        )
        
        return paperAffiliations.model_dump_json(indent = 1)
    

    def generateAsModel(self, input: str, tolerance= 5, paperId: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            try:
                output_object = self.outputClass(**json.loads(results))
            except:
                print(results)

        if (self.resultsfile and output_object):
            assert(id is not None), f"Found resultsfile but paperId parameter not passed."
            json_result = {paperId: output_object.model_dump()}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        else:
            json_result = {paperId: None}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        
        return output_object

