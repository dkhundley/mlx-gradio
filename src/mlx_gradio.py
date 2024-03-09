# Importing the necessary Python libraries
import os
import json
import gradio as gr



def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)



    

## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks() as demo:

    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    textbox.submit(greet, [chatbot, textbox], [chatbot])
    chatbot.like(vote, None, None)

    with gr.Row():
        text = gr.Textbox(label = 'David',
                          interactive = False)
        
    examples = gr.Examples(examples = ['David', 'Hundley'])
    



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Launching the Gradio Chatbot
    demo.launch()