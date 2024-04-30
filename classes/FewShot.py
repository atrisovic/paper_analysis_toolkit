from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from transformers import Pipeline
from pydantic import BaseModel as PydanticModel
from typing import Dict, List, Literal, Union
import json

def repair_brackets(s, depth = 1):
    return s.replace('{', '{{' * depth).replace('}', '}}' * depth)


class Contributor(PydanticModel):
    first: str
    last: str
    gender: Literal['male', 'female']
    
class Institution(PydanticModel):
    name: str
    type: Literal['academic', 'industry']
    
class PaperAffiliations(PydanticModel):
    contributors: List[Contributor]
    institutions: List[Institution]
    countries: List[str]

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
    
    
