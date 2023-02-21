import gradio as gr
import pickle
from utils.preprocessing import tokenize,reg_expressions
import numpy as np 

with open("models/current_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/current_bag_of_words","rb") as f:
    bag_of_words=pickle.load(f)
def preprocess_text(text):
    text=tokenize(text)
    text=reg_expressions(text)
    
    input_data = np.zeros((len(bag_of_words)))
    word_index = {word: i for i, word in enumerate(bag_of_words)}

    for word in text:
        if word in word_index:
            i = word_index[word]
            input_data[i] += 1
    input_data=input_data.reshape(1,len(bag_of_words))
    print(input_data.shape)
    return input_data

def classify_text(text):
    # Convert the text input into a format that the model expects
    input_data = preprocess_text(text)
    # Use the pre-trained model to generate a score for the input
    score = model.predict(input_data)
    # Classify the input as "good" or "bad" based on the score
    if score > 0.5:
        return "good review"
    else:
        return "bad review"



def prompt_input(prompt):
    return f"You entered: {prompt}"

inputs = gr.inputs.Textbox(lines=3, label="Enter your prompt")
outputs = gr.outputs.Textbox(label="Output")

gr.Interface(fn=classify_text, inputs=inputs, outputs=outputs, title="Sentiment Analysis of Amazon items", description="Enter the review and see if it's classified as 'good' or 'bad'.").launch()
