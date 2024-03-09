# Importing the necessary Python libraries
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setting constant values to represent model name and directory
MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
BASE_DIRECTORY = '../models/'

# Setting the full model directory path
model_directory = f'{BASE_DIRECTORY}{MODEL_NAME}'


# Checking to see if the directory has already been created
if os.path.exists(model_directory):

    # Loading the tokenizer and model from local file
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)

else:

    # Creating the new model directory
    os.makedirs(model_directory)

    # Downloading the tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Saving the tokenizer and model to model directory
    tokenizer.save_pretrained(save_directory = model_directory)
    model.save_pretrained(save_directory = model_directory)