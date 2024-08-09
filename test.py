import subprocess
from time import sleep
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser
from src.language_models.ChatInterface import LlamaCPPChatInterface

    
if __name__ == "__main__":
    model = LlamaCPPChatInterface(model_path = model_path)
    # Replace 'your_executable_path' with the actual path to your executable
    executable_path = '/home/gridsan/afogelson/llama.cpp/build/bin/llama-cli'
    model_path = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/saved_models/models--bullerwins--Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'
    interface = LlamaCPPChatInterface(executable_path, model_path)
    print(f"Output was: {interface.generate('What is 100x100?')}")