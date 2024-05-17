# Importing the necessary Python libraries
import os
import copy
import json
import uuid
import pandas as pd
import gradio as gr
import pandas as pd
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline



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



## GRADIO BACKEND LOAD
## ---------------------------------------------------------------------------------------------------------------------
def start_new_chat_history():
    '''
    Starts a new chat history if a local one cannot be found

    Inputs:
        - N/A

    Returns:
        - user_history (dict): A dictionary representing the user history that will be saved back to file
        - lc_user_conversation_history (dict): A LangChain managed version of the user's conversation history
    '''

    # Creating the user history from the base schema
    user_history = copy.deepcopy(BASE_USER_CONVERSATION_HISTORY_SCHEMA)

    # Instantiating an empty dictionary to hold the LangChain (LC) user conversation history
    lc_user_conversation_history = {}

    return user_history, lc_user_conversation_history



def load_chat_history_from_file(chat_history_json_location):
    '''
    Loads the chat history from file

    Inputs:
        - chat_history_json_location (str): The location of where the chat history JSON file resides

    Returns:
        - user_history (dict): A dictionary representing the user history that will be saved back to file
        - lc_user_conversation_history (dict): A LangChain managed version of the user's conversation history
    '''

    # Checking to see if any chat history exists
    if not os.path.exists(chat_history_json_location):
        
        # Instantiating a new user history if the chat does not exist
        user_history, lc_user_conversation_history = start_new_chat_history()

        return user_history, lc_user_conversation_history

    # Instantiating an empty dictionary to hold the LangChain (LC) user conversation history
    lc_user_conversation_history = {}

    # Loading the user history from the local JSON file
    with open(chat_history_json_location, 'r') as f:
        user_history = json.load(f)

    # Iterating over each conversation in the user history
    for conversation_id, conversation in user_history['chat_history'].items():
        
        # Instantiating a list to keep track of the chat interaction
        chat_interaction = []

        # Instantiating a LangChain chat message history object
        lc_chat_message_history = ChatMessageHistory()

        # Iterating over each chat interaction in the conversation history
        for chat_interaction in conversation['conversation']:

            # Appending any user messages as a LangChain HumanMessage
            if chat_interaction['role'] == 'user':
                lc_chat_message_history.add_message(HumanMessage(content = chat_interaction['content']))

            # Appending any assistant messages as a LangChain AIMessage
            if chat_interaction['role'] == 'assistant':
                lc_chat_message_history.add_message(AIMessage(content = chat_interaction['content'], metadata = chat_interaction['metadata']))

        # Appending the full conversation and metadata to the LC user conversation history with conversation ID as the key
        lc_user_conversation_history[conversation_id] = lc_chat_message_history

    return user_history, lc_user_conversation_history



def load_chat_history_as_df(user_history):
    '''
    Loads the chat history as a Pandas DataFrame for display purposes in the Gradio UI

    Inputs:
        - user_history (dict): A dictionary representing the user history that will be saved back to file

    Returns:
        - df_chat_history (Pandas DataFrame): A Pandas DataFrame that will be used as the reference point for displaying the chat history in the Gradio UI
        - df_reference (Pandas DataFrame): A Pandas DataFrame that will serve as the reference point between the display dataframe and the JSON chat history
    '''

    # Checking to see if there is any chat history
    if len(user_history['chat_history']) == 0:

        # Creating an empty DataFrame to represent no history
        df_chat_history = pd.DataFrame(data = {'Chat History': []})
        df_reference = pd.DataFrame(data = {'Chat History': [], 'Session ID': []})
        
        return df_chat_history, df_reference
    
    # Getting a list of all the summary titles
    summary_titles = {}
    for session_id, chat_entry in user_history['chat_history'].items():
        summary_titles[session_id] = chat_entry['summary_title']

    # Outputting the summary titles as a Pandas DataFrame
    df_chat_history = pd.DataFrame(data = {'Chat History': summary_titles.values()})
    df_reference = pd.DataFrame(data = {'Chat History': summary_titles.values(), 'Session ID': summary_titles.keys()})

    return df_chat_history, df_reference



## INITIAL LOAD VALUES
## ---------------------------------------------------------------------------------------------------------------------
# Loading the chat history from file
user_history, lc_user_conversation_history = load_chat_history_from_file(chat_history_json_location = chat_history_json_location)

# Forming the dataframe to represent the chat history in the UI
df_chat_history, df_reference = load_chat_history_as_df(user_history = user_history)

# Setting a new conversation ID to represent a new conversation
current_session_id = 'conv_id_' + str.replace(str(uuid.uuid4()), '-', '_')

# Loading the MLX model parameters
mlx_model_parameters = MLXModelParameters()

# Setting constant values to represent model name and directory
BASE_DIRECTORY = '../models'
MLX_DIRECTORY = f'{BASE_DIRECTORY}/mlx'
mlx_model_directory = f'{MLX_DIRECTORY}/{MODEL_NAME}'

# Setting up the LangChain MLX LLM
llm = MLXPipeline.from_model_id(
    model_id = mlx_model_directory,
    pipeline_kwargs = {
        'temp': mlx_model_parameters.temp,
        'max_tokens': mlx_model_parameters.max_tokens,
    }
)

# Setting up the LangChain MLX Chat Model with the LLM above
chat_model = ChatMLX(llm = llm)

# Setting up the Chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(messages = [
    SystemMessage(content = mlx_model_parameters.system_prompt),
    MessagesPlaceholder(variable_name = 'history'),
    HumanMessagePromptTemplate.from_template(template = '{input}')
])



## LANGCHAIN HELPER FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def correct_for_no_system_models(chat_messages):
    '''
    Precorrects for the issue where certain models (e.g. Llama, Mistral) are unable to accept for system messages

    Inputs:
        - chat_messages (LangChain ChatPromptValue): The current chat messages with no alterations

    Returns:
        - chat_messages (LangChain ChatPromptValue): The new chat messages with alterations (if needed)
    '''
    
    # Referencing MLX model parameters as a global object
    global mlx_model_parameters
    
    # Setting a list of models that we'll need to check against
    NO_SYSTEM_MODEL_PROVIDERS = ['mistralai', 'meta-llama']

    # Checking if the correction needs to be made if the model is Llama or Mistral
    if mlx_model_parameters.model_name.split('/')[0] in NO_SYSTEM_MODEL_PROVIDERS:

        # Getting the system message content
        system_message_content = chat_messages.messages[0].content

        # Replacing the System Message with a Human Message
        chat_messages.messages[0] = HumanMessage(content = system_message_content)

        # Adds a dummy AI Message
        chat_messages.messages.insert(1, AIMessage(content = ''))

    return chat_messages



def update_ai_response_metadata(ai_message):
    '''
    Updates the metadata on the AI response

    Inputs:
        - ai_message (LangChain AIMessage): The AI message produced by the model

    Returns:
        - ai_message (LangChain AIMessage): The AI message produced by the model, except now with the appropriate metadata intact
    '''

    # Referencing MLX model parameters as a global object
    global mlx_model_parameters

    # Creating a dictionary of the metadata that we will be adding to the AI message
    metadata = {
        'model_name': mlx_model_parameters.model_name,
        'timestamp': str(pd.Timestamp.utcnow()),
        'like_data': None,
        'hyperparameters': mlx_model_parameters.to_json()
    }

    # Applying the metadata to the AI response
    ai_message.response_metadata = metadata

    return ai_message



def generate_summary_title(lc_current_session_history, chat_model):
    '''
    Generates a summary title based on the user's initial interaction with the MLX model

    Inputs:
        - lc_current_session_history (LangChain ChatMessageHistory): A list of messages representing the current session conversation history
        - chat_model (LangChain MLX ChatModel): The chat model used to generate the summary title

    Returns:
        - summary_title (str): The summary title from the user's initial interaction with the MLX model
    '''

    # Referencing MLX model parameters as a global object
    global mlx_model_parameters

    # Creating the summary title prompt engineering
    summary_title_prompt = '''The text delineated by the triple backticks below contains the beginning of a conversation between a human and a large language model (LLM). Please provide a brief summary to serve as a title for this conversation. Do not use any system messages. Place more emphasis on the human's prompt over the AI's response. Please ensure the summary title does not exceed more than ten words. Please format the summary title as one would any formal title, like that of a book. Do not give any extra words except the summary title. (Example: Do not show "Title: ") Please ensure that the length of the output is no longer than 10 words.

    ```
    {history}
    ```
    '''

    # Creating the summary title chat prompt
    summary_title_chat_prompt = ChatPromptTemplate.from_messages(messages = [
        HumanMessagePromptTemplate.from_template(template = summary_title_prompt)
    ])

    # Creating a simple chain to produce the summary title
    summary_title_chain = summary_title_chat_prompt | chat_model
    
    # Getting just the first interaction from the chat message history
    initial_interaction = lc_current_session_history.messages[:2]

    # Generating the summary title based on the sample chat history
    summary_title_response = summary_title_chain.invoke({
        'history': initial_interaction
    })

    # Stripping out "Title: " (Because no idea why it wants to keep doing that...)
    summary_title = summary_title_response.content.replace("Title: ", "").replace("title: ", "")

    return summary_title



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    '''
    Gets the conversation session history per a uniquely input session ID

    Inputs:
        - session_id (str): A unique ID representing the current conversation session

    Returns:
        - lc_user_conversation_history[session_id] (LangChain BaseChatMessageHistory): A LangChain history object with the history of the conversation per the respective session ID
    '''
    if session_id not in lc_user_conversation_history:
        lc_user_conversation_history[session_id] = ChatMessageHistory()

    return lc_user_conversation_history[session_id]



def lc_session_to_json_session(lc_current_session_history):
    '''
    Converts the current LangChain session history to a JSON session history

    Inputs:
        - lc_current_session_history (LangChain ChatMessageHistory): The LangChain session history

    Returns:
        - json_current_session_history (list): The LangChain session history now transformed into JSON session history
    '''
    # Instantiating a list to hold the JSON outputs
    json_current_session_history = []

    # Converting the LangChain messages using LangChain's built in json() function
    lc_generated_json = json.loads(lc_current_session_history.json())

    # Iterating over each of the LangChain messages
    for message in lc_generated_json['messages']:

        # Determining the action based on if message is Human type
        if message['type'] == 'human':

            conversation_json = {
                'role': 'user',
                'content': message['content'],
            }
            json_current_session_history.append(conversation_json)
        
        # Determining the action based on if message is AI type
        elif message['type'] == 'ai':

            conversation_json = {
                'role': 'assistant',
                'content': message['content'],
                'metadata': message['response_metadata']
            }
            json_current_session_history.append(conversation_json)
        
    return json_current_session_history



## GRADIO BEHAVIOR FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def invoke_model(prompt_text, chatbot, chat_history):
    '''
    Invokes the model using MLX and saves chat history back to a local file

    Inputs:
        - prompt_text (str): The prompt text submitted by the user
        - chatbot (Gradio Chatbot): The Gradio chatbot to update
        - chat_history (Gradio DF): The Gradio dataframe object representing the chat history

    Returns
        - chatbot (Gradio Chatbot): The Gradio chatbot with the updated chat
    '''
    # Referencing global variables
    global lc_user_conversation_history
    global user_history
    global chat_history_json_location
    global chat_model
    global chat_prompt_template
    global mlx_model_parameters
    global current_session_id
    global df_chat_history
    global df_reference

    # Creating the inference chain by chaining together the chat prompt, chat model, and custom function to update metadata
    inference_chain = chat_prompt_template | RunnableLambda(correct_for_no_system_models) | chat_model | RunnableLambda(update_ai_response_metadata)

    # Creating a LangChain invokable that will run the inference pipeline and also manage chat history
    inference_chain_w_history = RunnableWithMessageHistory(
        runnable = inference_chain,
        get_session_history = get_session_history,
        input_messages_key = 'input',
        history_messages_key = 'history'
    )

    # Invoking the model with the user's input prompt and current session ID
    response = inference_chain_w_history.invoke(
        input = {'input': prompt_text},
        config = {
            'configurable': {
                'session_id': current_session_id
            }
        }
    )

    # Updating the user history with our demo conversation histories converted now to JSON
    user_history['chat_history'][current_session_id] = {
        'conversation': lc_session_to_json_session(lc_current_session_history = lc_user_conversation_history[current_session_id])
    }

    # If not added already, adding the system prompt to the current session
    if 'system_prompt' not in user_history['chat_history'][current_session_id].keys():
        user_history['chat_history'][current_session_id]['system_prompt'] = mlx_model_parameters.system_prompt

    # If not added already, adding the summary title to the current session
    if 'summary_title' not in user_history['chat_history'][current_session_id].keys():
        user_history['chat_history'][current_session_id]['summary_title'] = generate_summary_title(lc_current_session_history = lc_user_conversation_history[current_session_id], chat_model = chat_model)

        # Updating chat history in the UI
        df_chat_history, df_reference = load_chat_history_as_df(user_history = user_history)

        # Creating a UI element to hold a list of chat histories, listed by chat summary titles
        chat_history = gr.DataFrame(
            value = df_chat_history
        )

    # Writing the full history back to file
    with open(chat_history_json_location, 'w') as f:
        json.dump(user_history, f, indent = 4)

    # Updating the Gradio chatbot
    chatbot.append((prompt_text, response.content))

    # Clearing the prompt for the next user input
    prompt_text = ''

    return prompt_text, chatbot, chat_history



def load_existing_conversation(chatbot, selected_row: gr.SelectData):
    '''
    Loads an existing conversation from the conversation history list

    Inputs:
        - chatbot (Gradio chatbot): The Gradio chatbot object in its current state

    Returns:
        - chatbot (Gradio chatbot): The Gradio chatbot with the conversation history now loaded
    '''

    # Referencing global variables
    global current_session_id
    global df_chat_history
    global df_reference
    global user_history

    # Getting the index position for the reference table
    index_pos = selected_row.index[0]
    selected_session_id = df_reference.iloc[index_pos]['Session ID']

    # Loading the selected session ID's conversation history as a list of tuples
    loaded_conversation = []
    for i in range(0, len(user_history['chat_history'][selected_session_id]['conversation']), 2):
        user_interaction = user_history['chat_history'][selected_session_id]['conversation'][i]['content']
        ai_interaction = user_history['chat_history'][selected_session_id]['conversation'][i + 1]['content']
        loaded_conversation.append((user_interaction, ai_interaction))

    chatbot = gr.Chatbot(value = loaded_conversation)

    # Updating the current session ID
    current_session_id = selected_session_id

    return chatbot

    

def start_new_interaction(chatbot):
    '''
    Starts a new chat interaction

    Inputs:
        - chatbot (Gradio Chatbot): The UI element representing the chatbot

    Returns:
        - chatbot (Gradio Chatbot): The cleared out chatbot UI
    '''
    # Referencing current session ID as a global object
    global current_session_id

    # Clearing out the chatbot
    chatbot = gr.Chatbot(value = [])

    # Starting a new session ID
    current_session_id = 'conv_id_' + str.replace(str(uuid.uuid4()), '-', '_')

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
                    value = df_chat_history
                )

            # Setting the elements of the right part of the "Chat"
            with gr.Column(scale = 3):

                # Creating the chatbot user interaction UI
                chatbot = gr.Chatbot(label = 'Current Chat Interaction')

                # Creating another dummy row to columnize the user prompt vs. submit button
                with gr.Row():

                    # Creating the column to hold the user prompt textbox
                    with gr.Column(scale = 3):

                        # Creating the text box for the user to submit their prompt
                        prompt_text = gr.Textbox(
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
                

    # Defining the behavior for what occurs when the user hits "Enter" after typing a prompt
    prompt_text.submit(fn = invoke_model,
                       inputs = [prompt_text, chatbot, chat_history],
                       outputs = [prompt_text, chatbot, chat_history])

    # Defining the behavior for what occurs when the user clicks the arrow button to submit a prompt
    submit_prompt_button.click(fn = invoke_model,
                               inputs = [prompt_text, chatbot, chat_history],
                               outputs = [prompt_text, chatbot, chat_history])
    
    # Defining the behavior to load a historical conversation from the Gradio DataFrame object
    chat_history.select(fn = load_existing_conversation, inputs = [chatbot], outputs = [chatbot])

    # Defining the behavior to start a new conversation
    new_chat_button.click(fn = start_new_interaction, inputs = [chatbot], outputs = [chatbot])

    # chatbot.like(vote, None, None)
    



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Launching the Gradio Chatbot
    demo.launch(share = True)