PROMPT1 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. Please response in JSON format 
{{'classification': 'uses | extends | background | motivation | future_work | differences'}}, classifying the following sentence {input}"""


PROMPT2 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. If you don't feed that these categories are sufficient, you may respond with 'reject'. Please response in JSON format 
{{'classification': 'uses | extends | background | motivation | future_work | differences | reject'}}, classifying the following sentence {input}"""


PROMPT3 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model by looking at the sentences where it is cited. 
In particular, we want to classify these sentences with a boolean called "context". Context should be True if the citation is merely providing background, motivation, comparison, or some other relavant context within the sentence. 
If the model is used (the paper only deploys the model without modifications) or extended (if it suggests specific alterations to the model that change its structure or functioning), then the "context" if False, because more is being done than merely providing information.
Please response in this JSON format {{'context': 'True | False' }}, classifying the following sentence {input}"""


