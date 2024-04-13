# Importing the necessary Python libraries
import os
import json
import pandas as pd
import gradio as gr
from utils import *

test_data = {
    'Chat History': [
        'This is my first chat',
        'Here is another one',
        'And a third!'
    ],
    # 'Val': [
    #     '1',
    #     '2',
    #     '4'
    # ]
}

df = pd.DataFrame(test_data)


## CONSTANTS
## ---------------------------------------------------------------------------------------------------------------------
# Setting constant values to represent model name and directory
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
BASE_MODEL_DIRECTORY = '../models'
MLX_MODEL_DIRECTORY = f'{BASE_MODEL_DIRECTORY}/mlx'
mlx_model_directory = f'{MLX_MODEL_DIRECTORY}/{MODEL_NAME}'

# Setting a default meta prompt
DEFAULT_META_PROMPT = 'You are a helpful assistant.'


def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)


def load_historical_chat(row):
    print(row)
    pass



    

## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks(theme = gr.themes.Base()) as demo:

    with gr.Tab('Chat'):
        with gr.Row():
            with gr.Column(scale = 1):
                
                new_chat_button = gr.Button()

                chat_history = gr.DataFrame(
                    value = df
                )

            with gr.Column(scale = 4):
                chatbot = gr.Chatbot(show_label = False)
                textbox = gr.Textbox()
                textbox.submit(greet, [chatbot, textbox], [chatbot])
                chatbot.like(vote, None, None)

    with gr.Tab('Settings'):
        gr.Textbox()


    chat_history.select(fn = load_historical_chat, inputs = None, outputs = None)
    



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Loading the MLX model parameters
    mlx_model_parameters = MLXModelParameters()

    # Instantiating the MLX model using our quantized Mistral 7B
    mlx_model = MLXChatModel(mlx_path = mlx_model_directory)

    # Launching the Gradio Chatbot
    demo.launch()