
from pydantic import BaseModel as PydanticModel
from typing import List, Literal


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
