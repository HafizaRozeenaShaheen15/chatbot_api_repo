import json
import torch
import re
import os
import langid
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# Initialize models
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="./models/")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir="./models/")


similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
spell = SpellChecker()

# Load dataset
# with open('dataset.json') as f:
#     dataset = json.load(f)

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to dataset.json
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")

# Load dataset
with open(DATASET_PATH, 'r', encoding="utf-8") as f:
    dataset = json.load(f)
    

dataset_inputs = [item['input'].lower().strip() for item in dataset]
dataset_answers = [item['response'] for item in dataset]
dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# Helper functions
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def match_intent(user_input):
    corrected_input = correct_spelling(user_input)
    user_input_embedding = similarity_model.encode(corrected_input.lower().strip(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_input_embedding, dataset_embeddings)

    best_match_idx = torch.argmax(similarities).item()
    best_similarity_score = similarities[0][best_match_idx].item()

    return dataset_answers[best_match_idx] if best_similarity_score > 0.7 else None

def chatbot_response(user_input):
    if not user_input.strip():
        return "Please enter a valid question."

    lang = detect_language(user_input)
    if lang == 'ur':
        return "It seems you're asking in Roman Urdu. Please ask in English for now."

    matched_response = match_intent(user_input)
    return matched_response if matched_response else "I couldn't find an answer. Please try rephrasing your question."
