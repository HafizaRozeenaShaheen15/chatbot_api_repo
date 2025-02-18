# import json
# import torch
# import re
# import os
# import langid
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util
# from spellchecker import SpellChecker

# # Initialize models
# #  tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# #  model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="./models/")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir="./models/")


# similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# spell = SpellChecker()

# # Load dataset
# # with open('dataset.json') as f:
# #     dataset = json.load(f)

# # Get the absolute path of the current script
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Construct the full path to dataset.json
# DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")

# # Load dataset
# with open(DATASET_PATH, 'r', encoding="utf-8") as f:
#     dataset = json.load(f)
    

# dataset_inputs = [item['input'].lower().strip() for item in dataset]
# dataset_answers = [item['response'] for item in dataset]
# dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# # Helper functions
# def correct_spelling(text):
#     words = text.split()
#     corrected_words = [spell.correction(word) or word for word in words]
#     return ' '.join(corrected_words)

# def detect_language(text):
#     lang, _ = langid.classify(text)
#     return lang

# def match_intent(user_input):
#     corrected_input = correct_spelling(user_input)
#     user_input_embedding = similarity_model.encode(corrected_input.lower().strip(), convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(user_input_embedding, dataset_embeddings)

#     best_match_idx = torch.argmax(similarities).item()
#     best_similarity_score = similarities[0][best_match_idx].item()

#     return dataset_answers[best_match_idx] if best_similarity_score > 0.7 else None

# def chatbot_response(user_input):
#     if not user_input.strip():
#         return "Please enter a valid question."

#     lang = detect_language(user_input)
#     if lang == 'ur':
#         return "It seems you're asking in Roman Urdu. Please ask in English for now."

#     matched_response = match_intent(user_input)
#     return matched_response if matched_response else "I couldn't find an answer. Please try rephrasing your question."

import json
import torch
import langid
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# Initialize spell checker
spell = SpellChecker()

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set a unique pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load sentence transformer for intent matching
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load dataset
try:
    with open('dataset.json', encoding='utf-8') as f:
        dataset = json.load(f)
except FileNotFoundError:
    dataset = []

# Precompute embeddings for dataset questions
dataset_inputs = [item.get('input', '').lower().strip() for item in dataset]
dataset_answers = [item.get('response', '') for item in dataset]
dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# Maintain conversation history
conversation_history = []
answered_questions = {}

# Welcome message
def welcome_message():
    return ("🎓 Welcome to **Academic Navigator** – your university assistant! "
            "Ask about UOE admissions, scholarships, exams, and more. "
            "Type your question below or 'exit' to quit.")

# Normalize text (remove extra spaces)
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Detect language
def detect_language(user_input):
    lang, _ = langid.classify(user_input)
    
    # Additional check for Roman Urdu words
    roman_urdu_words = {"kya", "hai", "kar", "raha", "kaise", "mera", "tum", "ap"}
    if lang == 'ur' or any(word in user_input.lower().split() for word in roman_urdu_words):
        return "ur"
    
    return lang

# Correct spelling (fallback to original if no correction available)
def correct_spelling(user_input):
    words = user_input.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

# Match intent with sentence similarity
def match_intent(user_input, threshold=0.7):
    user_input_embedding = similarity_model.encode(user_input.lower().strip(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_input_embedding, dataset_embeddings)

    best_match_idx = torch.argmax(similarities).item()
    best_similarity_score = similarities[0][best_match_idx].item()

    return dataset_answers[best_match_idx] if best_similarity_score >= threshold else None

# Handle greetings
def handle_greeting(user_input):
    greetings = {"hello", "hi", "hey", "hye"}
    return "Hello! How can I assist you?" if user_input.lower() in greetings else None

# Handle off-topic questions
def handle_off_topic_questions(user_input):
    irrelevant_queries = {"how are you", "what are you doing", "what is your name"}
    return ("I'm here to assist with university queries. "
            "Ask about admissions, scholarships, exams, etc.") if user_input.lower() in irrelevant_queries else None

# Handle keyword-based queries
def handle_keyword_input(user_input):
    for item in dataset:
        if user_input in item['input'].lower():
            return item['response']
    return None

# Generate fallback response
def generate_fallback_response():
    return ("I'm not sure I understood that. Could you rephrase your question? "
            "I specialize in UOE-related topics like admissions, scholarships, and exams.")

# Main chatbot response function
def chatbot_response(user_input):
    user_input = normalize_text(user_input).lower()
    if not user_input:
        return "Please enter a valid question."

    # Maintain limited conversation history (keep last 10 messages)
    conversation_history.append(user_input)
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    # Detect language
    lang = detect_language(user_input)
    if lang == 'ur':
        return "It looks like you're using Roman Urdu. Please ask in English."

    # Handle greetings
    if greeting_response := handle_greeting(user_input):
        return greeting_response

    # Handle off-topic questions
    if off_topic_response := handle_off_topic_questions(user_input):
        return off_topic_response

    # Handle keyword-based input
    if keyword_response := handle_keyword_input(user_input):
        return keyword_response

    # Match intent
    corrected_input = correct_spelling(user_input)
    matched_response = match_intent(corrected_input)
    
    if matched_response:
        if user_input in answered_questions:
            return f"You asked that earlier! Here's a reminder: {answered_questions[user_input]}"
        answered_questions[user_input] = matched_response
        return matched_response

    # Default fallback response
    return generate_fallback_response()

# Example interaction loop
if __name__ == "__main__":
    print("Chatbot is starting...")
    print(welcome_message())

    while True:
        user_input = input("You: ").strip().lower()
        if user_input in {"exit", "bye", "quit"}:
            print("Chatbot: Thank you for chatting! Have a great day. 😊")
            break

        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
