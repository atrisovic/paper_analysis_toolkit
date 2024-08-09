from pydantic import BaseModel, confloat
import numpy as np
from typing import List, Optional
from src.language_models.ChatInterface import ChatInterface

class BoolAnswer(BaseModel):
    answer: bool 
    
class FloatAnswer(BaseModel):
    answer: confloat(ge = 0, le = 1) # type: ignore
    
    
class QuestionSet:
    def __init__(self, questions):
        self.questions = questions
        
    def get_answer_vector(self, response = List[Optional[BoolAnswer]], verbose = False):        
        answer_vector = np.vectorize(lambda s: s if s else 0)(np.array([None if not bool_answer else bool_answer.answer for bool_answer in response]))
        return answer_vector
    
    def ask_questions(self, subject, metadata, chat_interface: ChatInterface, prompt: str, tolerance = 1):
        results = []
        for question in self.questions:
            input = prompt.format(input = subject, modelKey = metadata, question = question)
            output = chat_interface.generateAsModel(input, tolerance = tolerance, strip_output = False)
            results.append(output)
        
        return results