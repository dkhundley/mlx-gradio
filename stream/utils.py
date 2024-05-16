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



## DEFAULT VALUES
## ---------------------------------------------------------------------------------------------------------------------
# Setting a default system prompt
DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

# Setting the default model 
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'



## MODEL PARAMETERS
## ---------------------------------------------------------------------------------------------------------------------
class MLXModelParameters():

    def __init__(self, model_name = MODEL_NAME, temp = 0.7, max_tokens = 1000, system_prompt = DEFAULT_SYSTEM_PROMPT):
        self.model_name = model_name
        self.temp = temp
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def __str__(self):
        return f'Model Name: {self.model_name}\nTemperature: {self.temp}\nMax Tokens: {self.max_tokens}\nSystem Prompt: {self.system_prompt}'
    
    def __repr__(self):
        return f'Model Name: {self.model_name}\nTemperature: {self.temp}\nMax Tokens: {self.max_tokens}\System Prompt: {self.system_prompt}'
    
    def update_model_name(self, new_model_name):
        self.model_name = new_model_name

    def update_temp(self, new_temp):
        self.temp = new_temp

    def update_max_tokens(self, new_max_tokens):
        self.max_tokens = new_max_tokens

    def update_system_prompt(self, new_system_prompt):
        self.system_prompt = new_system_prompt

    def to_json(self):
        return { 'temp': self.temp, 'max_tokens': self.max_tokens }
