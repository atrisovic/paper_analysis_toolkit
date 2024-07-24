from pydantic import BaseModel
from load_models import load_models
from src.language_models.ChatInterface import ChatInterface
from src.language_models.QuestionSet import QuestionSet
from datetime import datetime

class BoolAnswer(BaseModel):
    answer: bool
    
question_list = [
    "The CITING sentences mention results from the CITED paper to support background claims.",
    "The CITING sentences mention a technique of the CITED paper in order to provide relevant background context.",
    "The CITING sentences mention performance results of the CITED paper to contextualize the CITED model's capabilities.",
    "The CITING authors describe other researchers using the CITED foundation models.",
    "The CITING authors use the CITED foundation models to note a similarity or difference to an existing method.",
    "It's ambiguous as to whether or not the CITING authors actually use the CITED foundation model based on the wording.",
    "The CITING authors deploy the CITED foundation model's encoder or decoder as part of their methodology.",
    "The CITING authors use the CITED foundation model to create embeddings as part of their methodology.",
    "The CITING authors use the CITED foundation model for feature extraction as part of their methodology.",
    "The CITING authors use the CITED foundation model as a classifier or detector as part of their methodology.",
    "The CITING authors deploy the CITED foundation model to generate data in the form of text/image/audio/video for later training of their model.",
    "The CITING authors perform their own evaluation on the CITED foundation model.",
    "The CITING authors perform an ablation study on the CITED foundation model.",
    "The CITING authors clearly deploy the CITED foundation model at some point throughout their methodology.",
    "It's unclear whether or not the CITING authors perform any training on the CITED foundation model.",
    "The CITING authors deploy fine-tuning or adjusting the parameter weights of the CITED foundation model.",
    "The CITING authors pre-train or train-from-scratch the CITED model as part of their methodology.",
    "The CITING authors mention fine-tuning the CITED foundation model as a possibility, but do not mention fine-tuning themselves.",
    "The CITING authors train a model using the CITED paper's dataset.",
    "The CITING authors adopt the CITED foundation model's architecture as part of their model design.",
    "The CITING authors use the CITED foundation model to perform transfer learning.",
    "The CITING authors use the CITED foundation model as a benchmark for comparison.",
    "The CITING authors report improvements achieved by using the CITED foundation model over other models.",
    "The CITING authors integrate the CITED foundation model with other models or algorithms.",
    "The CITING authors use the CITED foundation model to validate a hypothesis or experimental setup.",
    "The CITING authors conduct a qualitative analysis involving the CITED foundation model.",
    "The CITING authors conduct a quantitative analysis involving the CITED foundation model.",
    "The CITING authors highlight limitations or challenges of using the CITED foundation model.",
    "The CITING authors discuss future work or potential extensions involving the CITED foundation model.",
    "The CITING authors leverage the CITED foundation model for a specific application domain (e.g., healthcare, finance, NLP).",
    "The CITING authors mention modifications or adaptations made to the CITED foundation model.",
    "The CITING authors deploy the CITED foundation model in a real-world scenario or experiment.",
    "The CITING authors use the CITED foundation model to perform anomaly detection.",
    "The CITING authors use the CITED foundation model to perform sentiment analysis.",
    "The CITING authors use the CITED foundation model for unsupervised learning tasks.",
    "The CITING authors use the CITED foundation model for supervised learning tasks.",
    "The CITING authors use the CITED foundation model for reinforcement learning tasks.",
    "The CITING authors use the CITED foundation model to process multi-modal data.",
    "The CITING authors use the CITED foundation model to enhance interpretability or explainability of their results.",
    "The CITING authors employ the CITED foundation model to optimize hyperparameters in their experiments.",
    "The CITING authors fine-tune the CITED foundation model on a domain-specific dataset.",
    "The CITING authors mention using a smaller learning rate specifically for fine-tuning the CITED foundation model.",
    "The CITING authors use the CITED foundation model's pre-trained weights as initialization for their own model.",
    "The CITING authors employ transfer learning techniques involving the CITED foundation model.",
    "The CITING authors use the CITED foundation model to pre-train a model before fine-tuning on a specific task.",
    "The CITING authors report using a specific dataset for fine-tuning the CITED foundation model.",
    "The CITING authors mention the number of epochs or iterations used for fine-tuning the CITED foundation model.",
    "The CITING authors discuss the computational resources needed for fine-tuning the CITED foundation model.",
    "The CITING authors mention specific hyperparameters adjusted during the fine-tuning of the CITED foundation model.",
    "The CITING authors compare results between pre-trained and fine-tuned versions of the CITED foundation model.",
    "The CITING authors mention the use of regularization techniques during the fine-tuning of the CITED foundation model.",
    "The CITING authors evaluate the CITED foundation model's performance before and after fine-tuning.",
    "The CITING authors describe modifying the architecture of the CITED foundation model prior to fine-tuning.",
    "The CITING authors report the impact of fine-tuning on the CITED foundation model's generalization ability.",
    "The CITING authors use the CITED foundation model in a semi-supervised learning framework involving fine-tuning.",
    "The CITING authors discuss challenges faced during the fine-tuning of the CITED foundation model.",
    "The CITING authors use data augmentation techniques in conjunction with fine-tuning the CITED foundation model.",
    "The CITING authors mention training the CITED foundation model on a multi-task learning setup.",
    "The CITING authors use the CITED foundation model to initialize another model which is then fine-tuned.",
    "The CITING authors highlight improvements in task performance due to fine-tuning the CITED foundation model."
][:20]

GENERIC_PROMPT = ("""The following sentences are from an academic paper (the CITING paper) which references a pretrained machine learning model through citation (the CITED paper). The models are called foundation models, and they might be a language model, a vision model, or any other kind of large neural network. The CITED paper is highlighed using HTML tags as such: <cite> cited reference </cite>. All other foundation models can be ignored, as we only care about the model cited with these tags. If it's helpful, the model identifier of the CITED model is {{modelKey}}. 

We'd like to discern how the CITING paper makes use of the foundation model, as described within the sentences. {question_statement}

\n
We want to be judicious and avoid guessing. The authors must explicitly mention the behavior in question specifically in relation to the CITED model with model identifier {{modelKey}}. Use only this JSON format in your response: {json_format}, based on the following:\n\n\"{{input}}\"""" )
SINGLEPROMPT = GENERIC_PROMPT.format(question_statement = "The following statement is to be evaluated as either true or false.\n{question}\n",
                                     json_format = '{{"answer": true | false}}')

model, tokenizer = load_models()
question_set = QuestionSet(questions = question_list)
interface = ChatInterface(model, tokenizer, device = 'cuda:0', outputClass = BoolAnswer)


subject = 'in this context, the main tool of interest, which popularized image synthesis, is called gan - generative adversarial network [107]. while alternatives to gans do exist, such as variational auto-encoders (vae) [151], normalizing flow (nf) techniques [228],[150], autoregressive models [293], and energy-based methods [118],[79], gans were typically at the lead in image generation. since their introduction and until recently, gans have undergone various improvements [222],<cite>[11]</cite>,[112],[327], and achieved stellar figure 7.1: visual comparison of deblurring results by pnp and red.'
modelKey = 'gans'

print(datetime.now())
answers = question_set.ask_questions(subject, metadata=modelKey, chat_interface=interface, prompt = SINGLEPROMPT)
vector = question_set.get_answer_vector(response = answers, verbose = True)
print(datetime.now())
print(answers, vector)