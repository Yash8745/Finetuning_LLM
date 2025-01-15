# app.py

from flask import Flask, request, jsonify
from transformers import pipeline
from model import load_model, tokenize_and_align_labels
from data_preprocessing import process_input

app = Flask(__name__)

# Load the fine-tuned model using pipeline
token_classifier = load_model()

@app.route('/')
def home():
    return "Welcome to the NER model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the user
        data = request.get_json()
        text = data['text']

        # Preprocess the input text
        processed_input = process_input(text)
        
        # Use the model to get predictions
        predictions = token_classifier(processed_input)

        # Return predictions as JSON
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
