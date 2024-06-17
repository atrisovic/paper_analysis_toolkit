PROMPT1 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. Please response in JSON format 
{{"classification": "uses | extends | background | motivation | future_work | differences"}}, classifying the following sentence {input}"""


PROMPT2 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. If you don't feed that these categories are sufficient, you may respond with 'reject'. Please response in JSON format 
{{"classification": "uses | extends | background | motivation | future_work | differences | reject"}}, classifying the following sentence {input}"""


PROMPT3 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model by looking at the sentences where it is cited. 
In particular, we want to classify these sentences with a boolean called "context". Context should be True if the citation is merely providing background, motivation, comparison, or some other relavant context within the sentence. 
If the model is used (the paper only deploys the model without modifications) or extended (if it suggests specific alterations to the model that change its structure or functioning), then the "context" if False, because more is being done than merely providing information.
Please response in this JSON format {{"context": "True | False" }}, classifying the following sentence {input}"""


PROMPT4 = """The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model by looking at the sentences where it is cited. 
In some cases, the citation is used merely as background (e.g. referencing the paper's results, using a technique but not the model, providing historical context, or similar).
In other cases, the citation indicates that the model itself is actually being used (e.g. accessed through an API, deploying the network, further enhancing or training through fine-tuning, or similar).
Please response in this JSON format: {{"classification": "context | uses" }}, classifying the following sentence {input}"""

PROMPT5 = """The following sentences are from an academic paper which references a foundation model through citation (a pretrained machine learning model for a particular task). This might be a language model, a vision model, or any other kind of large neural network. The CITING paper is the one from which we draw the sentences, and the CITED paper introduces the foundation model. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. 
We'd like to discern how the CITING paper makes use of the foundation model within the sentences. In particular, we'd like to classify whether or not the foundation model itself is actually used, or if the CITING paper is merely providing appropriate context using the CITED paper. 
For example, the following should be classified as context: the sentences reference the results of the CITED paper, the results of the CITED paper to provide relevant introductory background, a technique from the CITED paper, an architecture from the CITED paper, a loss function from the CITED paper, a dataset from the CITED paper, hyperparameters or optimization techniques from the CITED paper, displaying results or architectures from the CITED paper, etc. Do not limit responses just based on these examples.
Furthermore, the following might indicate the CITED model itself is being used: the CITING paper deploys the CITED model for classification; the CITING paper deploys the CITED model for generation, the CITING paper adapts the CITED model through fine-tuning, the CITING paper uses the CITED model to create embeddings, the CITING paper deploys the CITED model within a pipeline, the CITING paper uses the CITED model for feature extraction, etc. Do not limit responses just based on these examples.
Sometimes, the sentences are ambigous. Feel free to infer from the context of the sentence if there is a strong likelihood the rest of the paper would indicate "uses". There need not be an explicit reference to the model being used, but rather clear indications that this is likely.
Please response in this JSON format, including a one sentence explanation for your choice: {{"explanation": "The citation indicates..." , "classification": "context | uses" }}, classifying based on the following:\n {input}"""

PROMPT6 = """The following sentences are from an academic paper which references a foundation model through citation (a pretrained machine learning model for a particular task). This might be a language model, a vision model, or any other kind of large neural network. The CITING paper is the one from which we draw the sentences, and the CITED paper introduces the foundation model. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. 
We'd like to discern how the CITING paper makes use of the foundation model within the sentences. In particular, we'd like to classify whether or not the foundation model itself is actually used, or if the CITING paper is merely providing appropriate context using the CITED paper. 
For example, the following should be classified as context: the sentences reference the results of the CITED paper, the results of the CITED paper to provide relevant introductory background, a technique from the CITED paper, an architecture from the CITED paper, a loss function from the CITED paper, a dataset from the CITED paper, hyperparameters or optimization techniques from the CITED paper, displaying results or architectures from the CITED paper, etc.
Furthermore, the following might indicate the CITED model itself is being used: the CITING paper deploys the CITED model for classification; the CITING paper deploys the CITED model for generation, the CITING paper adapts the CITED model through fine-tuning, the CITING paper uses the CITED model to create embeddings, the CITING paper deploys the CITED model within a pipeline, the CITING paper uses the CITED model for feature extraction, etc.
Some of the sentences may be ambiguous. In the case of ambiguity, aire on the side of context rather than uses. Ideally all sentences classified as uses would be definitive cases of usage.
Please response in this JSON format, including a one sentence explanation for your choice: {{"explanation": "The citation indicates..." , "classification": "context | uses | unclear" }}, classifying based on the following:\n {input}"""


PROMPT7 = """The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>.

We'd like to discern how the CITING paper makes use of the foundation model within the sentences by classifying the citation as either "context" or "uses". By uses here, we mean using the model weights themselves, not the architecture, dataset, hyperparameters, or loss functions. We will think through this step-by-step:

1) Does the CITED model seem to be part of a methodology of the CITING paper, or is it being referenced as relevant background?
2) Is the CITED model being deployed in the methology (e.g. as a classifier, for generation of some kind, for fine-tuning, for embeddings, feature extraction, part of a pipeline, etc.)?
3) Is the CITED model being referenced to recap the history of research on a particular topic?
4) Is the CITED model used as a reference point in the methodology for comparison, but not for usage?

Using these answers as guidance, please response in this JSON format with two fields: an explanation including relevant answers to our step-by-step reasoning, and the final classification. The format is as follows: {{"explanation": "The citation indicates..." , "classification": "context | uses" }}, classifying based on the following:\n {input}""" 