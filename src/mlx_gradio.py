# Importing the necessary Python libraries
import os
import json
import pandas as pd
import gradio as gr
from utils import *

with open(file = '../data/schema.json') as f:
    test_json = json.load(f)
test_summaries = []
for entry in test_json['chat_history']:
    test_summaries.append(entry['summary_title'])
df = pd.DataFrame(data = {'Chat History': test_summaries})

# Loading the MLX model parameters
mlx_model_parameters = MLXModelParameters()



## MODEL DIRECTORY INFORMATION
## ---------------------------------------------------------------------------------------------------------------------
# Setting constant values to represent model name and directory
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
BASE_MODEL_DIRECTORY = '../models'
MLX_MODEL_DIRECTORY = f'{BASE_MODEL_DIRECTORY}/mlx'
mlx_model_directory = f'{MLX_MODEL_DIRECTORY}/{MODEL_NAME}'


def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)


def load_historical_chat(row_data: gr.SelectData, chatbot):
    chatbot = []
    dummy_user_prompt = f'Here is my prompt: {row_data.value}'
    dummy_completion = f'Oh, I like {row_data.value} too! It\'s the best!'

    chatbot.append((dummy_user_prompt, dummy_completion))
    chatbot.append((dummy_user_prompt, dummy_completion))
    return chatbot



    

## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks(theme = gr.themes.Base()) as demo:

    # Display a primary header label
    header_label = gr.Label(
        value = 'MLX Gradio Chat Interface',
        container = False
    )

    # Creating the interface for the "Chat" tab
    with gr.Tab('Chat'):

        # Creating a row just to create columns (required by Gradio to work this way)
        with gr.Row():

            # Setting the elements of the left part of the "Chat" tab
            with gr.Column(scale = 1):
                
                # Creating the "Start new interaction" button
                new_chat_button = gr.Button(
                    value = 'Start New Interaction',
                    variant = 'primary'
                )

                # Creating a UI element to hold a list of chat histories, listed by chat summary titles
                chat_history = gr.DataFrame(
                    value = df
                )

            # Setting the elements of the right part of the "Chat"
            with gr.Column(scale = 4):

                # Creating the chatbot user interaction UI
                chatbot = gr.Chatbot(label = 'Current Chat Interaction')

                # Creating another dummy row to columnize the user prompt vs. submit button
                with gr.Row():

                    # Creating the column to hold the user prompt textbox
                    with gr.Column(scale = 3):

                        # Creating the text box for the user to submit their prompt
                        user_prompt = gr.Textbox(
                            placeholder = 'Ask anything...',
                            show_label = False
                        )

                    # Creating the column to hold the user prompt submission button
                    with gr.Column(scale = 1):

                        # Creating the button for the user to submit their prompt
                        submit_prompt_button = gr.Button(
                            value = '↑',
                            scale = 1
                        )

    # Creating the interface for the "Settings" tab
    with gr.Tab('Settings'):

        # Creating a textbox for the user to update the system
        system_prompt = gr.Textbox(
            label = 'System Prompt (Note: System prompts are only applied to new chat interactions.)',
            value = mlx_model_parameters.system_prompt,
            interactive = True
        )

        # Creating a row to be able to put sliders side-by-side
        with gr.Row():
            
            # Creating the column to hold the temperature slider
            with gr.Column():

                # Creating a slider to change the temperature
                temp_slider = gr.Slider(
                    label = 'Temperature',
                    minimum = 0,
                    maximum = 2,
                    value = mlx_model_parameters.temp,
                    interactive = True
                )

            # Creating the column to hold the max tokens slider
            with gr.Column():

                # Creating a solider to change the maximum number of tokens
                max_tokens_slider = gr.Slider(
                    label = 'Maximum Tokens',
                    minimum = 1,
                    maximum = 10000,
                    value = mlx_model_parameters.max_tokens,
                    interactive = True
                )
                


    chat_history.select(fn = load_historical_chat, inputs = [chatbot], outputs = [chatbot])
    chatbot.like(vote, None, None)
    



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Instantiating the MLX model using our quantized Mistral 7B
    # mlx_model = MLXChatModel(mlx_path = mlx_model_directory)

    # Launching the Gradio Chatbot
    demo.launch(share = True)