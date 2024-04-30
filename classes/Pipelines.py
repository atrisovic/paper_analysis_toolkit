from pydantic import BaseModel as PydanticModel
from classes.Affiliations import PaperAffiliations, Contributor, Institution
from transformers import Pipeline
import logging, json
from typing import Dict, List, Union
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from transformers import Pipeline


def repair_brackets(s, depth = 1):
    return s.replace('{', '{{' * depth).replace('}', '}}' * depth)


logger = logging.getLogger(__name__)


class FewShotExample(PydanticModel):
    question: str
    answer: str
    
class FewShotPipeline:
    pipeline: Pipeline
    examples: List[Dict]
    
    
    def __init__(self, pipeline: Pipeline, outputClass: PydanticModel = PaperAffiliations):
        self.pipeline = pipeline
        self.examples = []
        self.outputClass = outputClass
        
    def addExample(self, question: str, answer: Union[str, PydanticModel]):
        if (isinstance(answer, PydanticModel)):
            example = FewShotExample(question = repair_brackets(question), answer = repair_brackets(answer.model_dump_json(indent=1)))
        else:
            example = FewShotExample(question = repair_brackets(question), answer = repair_brackets(answer))
            
        self.examples.append(example)   
            
    def getPromptTemplater(self, max_examples: int = None) -> str:
        schema = json.dumps(self.outputClass.model_json_schema()['$defs'], indent=1)
             
        template= f"""Please read this text, and return the following information in the JSON format provided: \n
                   {repair_brackets(schema, depth = 2)}\n
                    The output should match exactly the JSON format given. The text is as follows {{question}}:\n JSON:\n{{answer}}"""
                                
        examples = [example.model_dump() for example in self.examples][:max_examples]
        
        suffix=f"""Please read this text, and return the following information in the JSON format provided: \n 
                   {repair_brackets(schema)}
                    \n The output should match exactly the JSON format given. The text is as follows {input}:\n JSON:\n"""
        
        
        
        example_prompt = PromptTemplate(
                    input_variables=["question", "answer"], 
                    template = template
                )             
   
        prompt_templater = FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=example_prompt,
                    suffix=suffix,
                    input_variables=["input"],
                )
        
        return prompt_templater
    
    def generate(self, input: str, max_examples: int = None):
        prompt_template = self.getPromptTemplater(max_examples = max_examples)
        
        chain = prompt_template | self.pipeline
        result = chain.invoke({'input': input})
        return result
    
class AffiliationsPipeline(FewShotPipeline):
    def __init__(self, pipeline: Pipeline, outputClass: PydanticModel = PaperAffiliations, resultsfile: str = None):
        super().__init__(pipeline = pipeline, outputClass=outputClass)
        
        
        self.resultsfile = resultsfile
        
        for question, answer in self.getExamples():
            self.addExample(question=question, answer=answer)


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
    

    def generateAsModel(self, input: str, tolerance=1, paperId: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            try:
                output_object = self.outputClass(**json.loads(results))
            except:
                pass

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

