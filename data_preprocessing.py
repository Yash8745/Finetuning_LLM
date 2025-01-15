from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def process_input(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, is_split_into_words=False)
    return inputs
