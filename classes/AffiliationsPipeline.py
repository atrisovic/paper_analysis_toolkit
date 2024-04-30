from pydantic import BaseModel as PydanticModel
from classes.FewShot import FewShotPipeline, PaperAffiliations, Contributor, Institution
from transformers import Pipeline
import logging, json


logger = logging.getLogger(__name__)


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
        
        return output_object

