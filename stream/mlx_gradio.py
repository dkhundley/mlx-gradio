import gradio as gr
import json
import pandas as pd

with open(file = '../data/schema.json') as f:
    test_json = json.load(f)
test_summaries = []
for entry in test_json['chat_history']:
    test_summaries.append(entry['summary_title'])
df = pd.DataFrame(data = {'Chat History': test_summaries})


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