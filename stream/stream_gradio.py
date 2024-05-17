import json
import gradio as gr
import pandas as pd

with open(file = '../data/schema.json') as f:
    test_json = json.load(f)
test_summaries = []
for entry in test_json['chat_history']:
    test_summaries.append(entry['summary_title'])
df = pd.DataFrame(data = {'Chat History': test_summaries})



## DEFAULT VALUES
## ---------------------------------------------------------------------------------------------------------------------
# Setting a default system prompt
DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

# Setting the default model 
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'

# Setting the default conversation history schema
BASE_USER_CONVERSATION_HISTORY_SCHEMA = {
    'user_id': 'default_username',
    'chat_history': {}
}

# Setting the default chat history lcoation
chat_history_json_location = '../data/chat_history.json'



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



## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
with gr.Blocks(theme = gr.themes.Base()) as demo:

    header_label = gr.Label(
        value = 'MLX Gradio Chat Interface',
        container = False
    )

    with gr.Tab('Chat'):

        with gr.Row():

            with gr.Column(scale = 1):

                new_chat_button = gr.Button(
                    value = 'Start New Interaction',
                    variant = 'primary'
                )

                chat_history = gr.DataFrame(
                    value = df
                )

            with gr.Column(scale = 3):

                chatbot = gr.Chatbot(label = 'Current Chat Interaction')

                with gr.Row():
                    
                    with gr.Column(scale = 4):
                        user_prompt = gr.Textbox(
                            placeholder = 'Ask anything...',
                            show_label = False
                        )
                    
                    with gr.Column(scale = 1):
                        
                        submit_prompt_button = gr.Button(
                            value = '↑',
                            scale = 1
                        )

    with gr.Tab('Settings'):

        system_prompt = gr.Textbox(
            label = 'System Prompt (Note: System prompts are only applied to new chat interactions.)',
            value = 'You are a helpful assistant.',
            interactive = True
        )

        with gr.Row():

            with gr.Column():

                temp_slider = gr.Slider(
                    label = 'Temperature',
                    minimum = 0,
                    maximum = 2,
                    value = 0.7,
                    interactive = True
                )

            with gr.Column():

                max_tokens_slider = gr.Slider(
                    label = 'Max Tokens',
                    minimum = 1,
                    maximum = 100000,
                    value = 10000,
                    interactive = True
                )



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Launching the Gradio Chatbot
    demo.launch(share = True)