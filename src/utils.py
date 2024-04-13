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



# Setting constant values to represent model name and directory
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
BASE_MODEL_DIRECTORY = '../models'
MLX_MODEL_DIRECTORY = f'{BASE_MODEL_DIRECTORY}/mlx'
mlx_model_directory = f'{MLX_MODEL_DIRECTORY}/{MODEL_NAME}'



## MODEL PARAMETERS
## ---------------------------------------------------------------------------------------------------------------------
class MLXModelParameters():

    def __init__(self, temp = 0.7, max_tokens = 1000):
        self.temp = temp
        self.max_tokens = max_tokens

    def __str__(self):
        return f'Temperature: {self.temp}\nMax Tokens: {self.max_tokens}'
    
    def __repr__(self):
        return f'Temperature: {self.temp}\nMax Tokens: {self.max_tokens}'

    def update_temp(self, new_temp):
        self.temp = new_temp

    def update_max_tokens(self, new_max_tokens):
        self.max_tokens = new_max_tokens

    def to_json(self):
        return { 'temp': self.temp, 'max_tokens': self.max_tokens }


def load_model():
    '''
    Loads MLX model & tokenizer

    Inputs:
        - ?
    
    Returns:
        - ?
    '''


def load_chat_history():
    '''
    Loads chat history from file

    Inputs:
        - ?
    
    Returns:
        - ?
    '''
    pass



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