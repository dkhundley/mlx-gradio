# Importing the necessary Python libraries
import os
import json
import pandas as pd
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import root_validator, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult



## LANGCHAIN SETUP
## ---------------------------------------------------------------------------------------------------------------------
# Setting a default meta prompt
DEFAULT_META_PROMPT = 'You are a helpful assistant.'



## MODEL PARAMETERS
## ---------------------------------------------------------------------------------------------------------------------
class MLXModelParameters():

    def __init__(self, temp = 0.7, max_tokens = 1000, meta_prompt = DEFAULT_META_PROMPT):
        self.temp = temp
        self.max_tokens = max_tokens
        self.meta_prompt = meta_prompt

    def __str__(self):
        return f'Temperature: {self.temp}\nMax Tokens: {self.max_tokens}\nMeta Prompt: {self.meta_prompt}'
    
    def __repr__(self):
        return f'Temperature: {self.temp}\nMax Tokens: {self.max_tokens}\nMeta Prompt: {self.meta_prompt}'

    def update_temp(self, new_temp):
        self.temp = new_temp

    def update_max_tokens(self, new_max_tokens):
        self.max_tokens = new_max_tokens

    def update_meta_prompt(self, new_meta_prompt):
        self.meta_prompt = new_meta_prompt

    def to_json(self):
        return { 'temp': self.temp, 'max_tokens': self.max_tokens }
    


## MLX LANGCHAIN CUSTOM INTEGRATION
## ---------------------------------------------------------------------------------------------------------------------
# # Instantiating the class representing the MLX Chat Model, inheriting from LangChain's BaseChatModel
# class MLXChatModel(BaseChatModel):
#     mlx_path: str
#     mlx_model: Any = Field(default = None, exclude = True)
#     mlx_tokenizer: Any = Field(default = None, exclude = True)
#     max_tokens: int = Field(default = 1000)

#     @property
#     def _llm_type(self) -> str:
#         return 'MLXChatModel'
    
#     @root_validator()
#     def load_model(cls, values: Dict) -> Dict:

#         # Loading the model and tokenizer with the input string
#         model, tokenizer = mlx_load(path_or_hf_repo = values['mlx_path'])
        
#         # Saving the variables back appropriately
#         values['mlx_model'] = model
#         values['mlx_tokenizer'] = tokenizer
#         return values
    
#     def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> ChatResult:

#         # Instantiating an empty string to represent the prompt we will be generating in the end
#         prompt = ''

#         # Extracting the raw text from each of the LangChain message types
#         for message in messages:
#             prompt += f'\n\n{message.content}'

#         # Generating the LLM response using MLX
#         mlx_response = mlx_generate(
#             model = self.mlx_model,
#             tokenizer = self.mlx_tokenizer,
#             max_tokens = self.max_tokens,
#             prompt = prompt
#         )

#         # Returning the MLX response as a proper LangChain ChatResult object
#         return ChatResult(generations = [ChatGeneration(message = AIMessage(content = mlx_response))])



## GRADIO BACKEND LOAD
## ---------------------------------------------------------------------------------------------------------------------
def load_chat_history():
    '''
    Loads chat history from file

    Inputs:
        - ?
    
    Returns:
        - ?
    '''
    pass



## GRADIO INTERACTIONS
## ---------------------------------------------------------------------------------------------------------------------
def load_chat_interaction():
    '''
    Loads a specific chat interaction from the user's history

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass


def process_user_prompt():
    '''
    Takes in a user's prompt to generate an answer (completion) from the AI

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass


def generate_summary_title():
    '''
    Generates a summary title based on the interaction between the user and the AI

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass



def save_chat_history():
    '''
    Saves chat history to file

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass



def start_new_chat():
    '''
    Starts a new chat interaction

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass