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


PROMPT8 = """The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. All other citations can be ignored, as we only care about the cited reference with these tags.

We'd like to discern how the CITING paper makes use of the foundation model within the sentences. Below are ten statements, which will be evaluated as either true or false.

1. The CITING sentences use the results of the CITED paper to support background claims.
2. The CITING sentences use the techniques of the CITED paper to provide relevant context.
3. The CITING sentences display performance results of the CITED paper.
4. The CITING paper uses the CITED foundation model's encoder or decoder.
5. The CITING paper uses the CITED foundation model to create embeddings.
6. The CITING paper further fine-tuned or adjusts the weights of the CITED foundation model. 
7. The CITING paper recreates the CITED foundation model's architecture.
8. The CITING paper trains a model using the CITED paper's dataset.
9. The CITING paper uses the CITED foundation model for feature extraction.
10. The CITING paper uses the CITED foundation model as a classifier.
11. The CITING paper uses the CITED foundation model to generate text/image/audio/video samples.
12. The CITING paper uses the CITED foundation model as a backbone model.


Please response in this JSON format: {{"1": "True | False" , "2": "True | False", "3": "True | False", ... , "11": "True | False"}}, classifying based on the following:\n {input}""" 



GENERIC_PROMPT = ("""The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. All other foundation models can be ignored, as we only care about the model cited with these tags. If it's helpful, the model identifier of the CITED model is {{modelKey}}. 

We'd like to discern how the CITING paper makes use of the foundation model, as described within the sentences. {question_statement}

\n
We want to be judicious and avoid guessing. The authors must explicitly mention the behavior in question specifically in relation to the CITED model with model identifier {{modelKey}}. Use only this JSON format in your response: {json_format}, based on the following:\n\n\"{{input}}\"""" )

SINGLEPROMPT = GENERIC_PROMPT.format(question_statement = "The following statement is to be evaluated as either true or false.\n{question}\n",
                                     json_format = '{{"answer": true | false}}')


GENERIC_PROMPT_COT = ("""The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. All other foundation models can be ignored, as we only care about the model cited with these tags. If it's helpful, the model identifier of the CITED model is {{modelKey}}. 

We'd like to answer determine whether the following statement is true or false:
{question_statement}

\n
Use only this JSON format in your response: {json_format}. The sentences are as follows:\n\n\"{{input}}\"""" )

SINGLEPROMPT_COT = GENERIC_PROMPT_COT.format(question_statement = "First, think out loud and explain your thoughts step-by-step in one sentence, then give a true/false answer.\n{question}\n",
                                     json_format = '{{"explanation": "Step-by-step thought process here", "answer": true | false}}')


GENERIC_PROMPT_COT_CONFIDENCE = ("""The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. All other foundation models can be ignored, as we only care about the model cited with these tags. If it's helpful, the model identifier of the CITED model is {{modelKey}}. 

We'd like to answer determine how likely the statement below is to be true using a confidence value between 0 and 1. For example, almost certainly false should be 0, almost certainly true should be 1, and complete ambiguity is 0.5. You may give any value between 0 and 1.
{question_statement}

\n
Use only this JSON format in your response: {json_format}. The sentences are as follows:\n\n\"{{input}}\"""" )

SINGLEPROMPT_COT_CONFIDENCE = GENERIC_PROMPT_COT_CONFIDENCE.format(question_statement = "First, think out loud and explain your thoughts step-by-step in one sentence, then give a confidence score as your final answer.\n{question}\n",
                                     json_format = '{{"explanation": "Step-by-step thought process here", "answer": float}}')
