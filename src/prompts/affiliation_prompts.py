PROMPT1 = """Please read this text (in .mmd format), and return the following information in the JSON format provided: \n{schema}\n The output should match exactly the JSON format below. Determine the country only if explicitly mentioned, or implicit in the institutions, not from the names. The text is as follows:\n\n {input}""" 
PROMPT2 = """Please read this text (in .mmd format), and return the following information in the JSON format provided: \n{schema}\n The output should match exactly the JSON format below. Determine the country only if explicitly mentioned, or implicit in the institutions, not from the names. Pick the best institution type as either "academic", "industry"; DO NOT EVER include use other labels. The text is as follows:\n\n {input}""" 
PROMPT3 = """The following text is a sample of markdown extracted from a research paper. I'd like to know the relevant institutions who authored this paper, if that information is in the sample.
Please read this text (in .mmd format), and return the following information in the JSON format provided: \n{schema}
The output should match exactly the JSON format below. Determine the country only if explicitly mentioned, or implicit in the institutions, not from the names. 
Pick the best institution type as either "academic", "industry"; DO NOT EVER include use other labels. 
The text is as follows:\n\n {input}""" 
