# Importing the necessary Python libraries
import os
import json
import uuid
import copy
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



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
    '''

    # Checking to see if there is any chat history
    if len(user_history['chat_history']) == 0:

        # Creating an empty DataFrame to represent no history
        df_chat_history = pd.DataFrame(data = {'Chat History': []})
        
        return df_chat_history
    
    # Getting a list of all the summary titles
    summary_titles = []
    for chat_entry in user_history['chat_history'].values():
        summary_titles.append(chat_entry['summary_title'])

    # Outputting the summary titles as a Pandas DataFrame
    df_chat_history = pd.DataFrame(data = {'Chat History': summary_titles})

    return df_chat_history




## GRADIO PARAMETER UPDATES
## ---------------------------------------------------------------------------------------------------------------------
def update_system_prompt():
    '''
    Updates the system prompt with a new one per the user's input

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass



def update_temperature():
    '''
    Updates the temperature with a new one per the user's input

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass



def update_max_tokens():
    '''
    Updates the system prompt with a new one per the user's input

    Inputs:
        - ?

    Returns:
        - ?
    '''
    pass



## GRADIO INTERACTIONS
## ---------------------------------------------------------------------------------------------------------------------
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