import gradio as gr

# Initialize chat history and titles
chat_histories = []
chat_titles = []
current_chat_index = -1

def chatbot(input_text):
    global current_chat_index
    if current_chat_index == -1:
        # Start a new chat
        chat_histories.append([])
        chat_titles.append(generate_title(input_text))
        current_chat_index = len(chat_histories) - 1
    
    # Your chatbot logic here
    response = generate_response(input_text)
    chat_histories[current_chat_index].append((input_text, response))
    return response, chat_histories[current_chat_index], chat_titles

def generate_response(input_text):
    # Placeholder function for generating chatbot responses
    return "This is a sample response to: " + input_text

def generate_title(input_text):
    # Placeholder function for generating chat titles
    return "Chat about: " + input_text[:20] + "..."

def load_chat(title):
    global current_chat_index
    if title in chat_titles:
        current_chat_index = chat_titles.index(title)
        return chat_histories[current_chat_index]
    else:
        return []

def new_chat():
    global current_chat_index
    current_chat_index = -1
    return [], []

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot with Chat History")
    
    with gr.Row():
        with gr.Column():
            chat_list = gr.Dataframe(headers=["Chat Titles"], datatype=["str"], col_count=(1, "fixed"))
            new_chat_btn = gr.Button("New Chat")
        
        with gr.Column():
            chatbot_output = gr.Chatbot(label="Chatbot")
            input_text = gr.Textbox(placeholder="Enter your message...")
            submit_btn = gr.Button("Send")
    
    chat_list.select(load_chat, inputs=chat_list, outputs=chatbot_output)
    new_chat_btn.click(new_chat, outputs=[chatbot_output, chat_list])
    submit_btn.click(chatbot, inputs=input_text, outputs=[chatbot_output, chatbot_output, chat_list])
    
    gr.Examples(
        examples=["Hello!", "How are you?", "Tell me a joke."],
        inputs=input_text
    )

demo.launch()