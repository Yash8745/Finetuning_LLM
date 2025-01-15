# model.py

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def load_model():
    # Load the fine-tuned model checkpoint
    checkpoint = "notebooks/distilbert-finetuned-ner/checkpoint-5268"  # Path to your fine-tuned model
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Set up the pipeline for token classification
    token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return token_classifier
