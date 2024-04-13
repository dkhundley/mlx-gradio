import gradio as gr
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)

# Function to handle the selected row
def process_selected_row(evt: gr.SelectData):
    if evt.index is None:
        return "No row selected"
    
    # Perform the desired action with the selected row data
    row_data = df.iloc[evt.index[0]]
    name = row_data['Name']
    age = row_data['Age']
    city = row_data['City']
    result = f"Selected row: Name={name}, Age={age}, City={city}"
    return result

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Select a row from the DataFrame:")
    
    # Create a DataFrame component
    dataframe = gr.DataFrame(df)
    
    # Create an output component to display the result
    output = gr.Textbox()
    
    # Attach the select event listener to the DataFrame
    dataframe.select(process_selected_row, None, output)

# Launch the Gradio interface
demo.launch()
