# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util
# import torch
# import langid  # For language detection
# import re
# import os
# from spellchecker import SpellChecker  # For spelling correction

# # Initialize the spell checker
# spell = SpellChecker()

# # Load the DialoGPT medium model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # Set a unique pad token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Load sentence transformer model for intent matching
# similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Load dataset for university-specific information
# # with open('dataset.json') as f:
# #     dataset = json.load(f)


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, 'dataset.json')

# # Load dataset
# with open(DATASET_PATH, encoding='utf-8') as f:
#     dataset = json.load(f)


# # Precompute embeddings for dataset questions
# dataset_inputs = [item['input'].lower().strip() for item in dataset]
# #dataset_inputs = [item.get('input', '').lower().strip() for item in dataset]

# dataset_answers = [item['response'] for item in dataset]
# dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# # dataset_inputs = [item.get('input', '').lower().strip() for item in dataset]
# # dataset_answers = [item.get('response', '') for item in dataset]  # Using `get()` to avoid KeyError
# # dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# # Maintain conversation history for context
# conversation_history = []
# answered_questions = {}  # Store answered questions

# # Introduction message when the chatbot starts
# def welcome_message():
#     return ("Hello! 🎓 Welcome to **Academic Navigator** –  the university information assistant for UOE.  "
#             "I have the answers to your university-related questions, such as admissions, scholarships, exams, and more. "
#             "How can I assist you today? "
#             "Type your question below or 'exit' to quit. ")

# # Normalize spacing by replacing multiple spaces with a single space
# def normalize_spacing(text):
#     return re.sub(r'\s+', ' ', text).strip()

# # Handle greeting messages like "hello" or "hye"
# def handle_greeting(user_input):
#     greetings = ["hello", "hye", "hi"]
#     if user_input.lower() in greetings:
#         return "Hello! How can I assist you with university-related questions today?"

# # Predefined responses for off-topic or irrelevant questions
# def handle_off_topic_questions(user_input):
#     irrelevant_queries = ["how are you", "what are you doing", "what is your name"]
#     if user_input.lower() in irrelevant_queries:
#         return ("I'm your university information assistant, here to help with UOE-related queries. "
#                 "Please ask me about university topics such as admissions, scholarships, exams, and more.")

# # Maintain conversation context
# def maintain_conversation_context(user_input):
#     conversation_history.append(user_input)

# # Detect language (Roman Urdu or English)
# def detect_language(user_input):
#     lang, _ = langid.classify(user_input)
#     return lang

# # Clean the input before spelling correction
# def clean_input(user_input):
#     return re.sub(r'[^a-zA-Z0-9\s]', '', user_input)

# # Correct spelling mistakes using SpellChecker
# def correct_spelling(user_input):
#     user_input = clean_input(user_input.lower().strip())  # Clean input
#     words = user_input.split()
#     corrected_words = [spell.correction(word) for word in words]
#     return ' '.join(corrected_words)

# # Match user input with dataset using similarity
# def match_intent(user_input, threshold_low=0.5, threshold_high=0.7):
#     corrected_input = correct_spelling(user_input)
#     user_input_embedding = similarity_model.encode(corrected_input.lower().strip(), convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(user_input_embedding, dataset_embeddings)

#     best_match_idx = torch.argmax(similarities).item()
#     best_similarity_score = similarities[0][best_match_idx].item()

#     if best_similarity_score > threshold_high:
#         return dataset[best_match_idx]['response']
#     elif best_similarity_score > threshold_low:
#         return dataset[best_match_idx]['response']
#     return None

# # Handle single-word or keyword-based input
# def handle_keyword_input(user_input):
#     user_input = user_input.lower().strip()
#     for item in dataset:
#         if user_input in item['input'].lower():
#             return item['response']
#     return None

# # Generate fallback response
# def generate_fallback_response():
#     return ("Could you please rephrase? "
#             "I specialize in UOE-related information. Please ask about university topics like admissions, scholarships, exams, etc.")

# # Main chatbot response function
# def chatbot_response(user_input):
#     user_input = normalize_spacing(user_input)
#     maintain_conversation_context(user_input)

#     lang = detect_language(user_input)
#     if lang == 'ur':
#         return "It seems you're asking in Roman Urdu. How can I assist you with UOE-related queries?"

#     greeting_response = handle_greeting(user_input)
#     if greeting_response:
#         return greeting_response

#     off_topic_response = handle_off_topic_questions(user_input)
#     if off_topic_response:
#         return off_topic_response

#     keyword_response = handle_keyword_input(user_input)
#     if keyword_response:
#         return keyword_response

#     matched_response = match_intent(user_input)
#     if matched_response:
#         if user_input in answered_questions:
#             return f"I've already answered that: {answered_questions[user_input]}"
#         answered_questions[user_input] = matched_response
#         return matched_response

#     return generate_fallback_response()

# # Example interaction loop
# if __name__ == "__main__":
#     print("Chatbot is starting...")
#     print(welcome_message())

#     while True:
#         user_input = input("You: ").strip().lower()
#         if user_input in {"exit", "bye", "quit"}:
#             print("Chatbot: Thank you for chatting! Have a great day. 😊")
#             break

#         response = chatbot_response(user_input)
#         print(f"Chatbot: {response}")




import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import langid  
import os
import re
from spellchecker import SpellChecker  # For spelling correction

# Initialize the spell checker
spell = SpellChecker()

# Load the DialoGPT medium model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set a unique pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load sentence transformer model for intent matching
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load dataset for university-specific information
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.json')

# Load dataset
with open(DATASET_PATH, encoding='utf-8') as f:
    dataset = json.load(f)

# Precompute embeddings for dataset questions
dataset_inputs = [item.get('input', '').lower().strip() for item in dataset]
dataset_answers = [item.get('response', '') for item in dataset]
dataset_embeddings = similarity_model.encode(dataset_inputs, convert_to_tensor=True)

# Maintain conversation history for context
conversation_history = []
answered_questions = {}  # Store answered questions

# Normalize spacing by replacing multiple spaces with a single space
def normalize_spacing(text):
    return re.sub(r'\s+', ' ', text).strip()

# Handle greeting messages like "hello" or "hi"
def handle_greeting(user_input):
    greetings = ["hello", "hye", "hi"]
    if user_input.lower() in greetings:
        return "Hello! How can I assist you with university-related questions today?"

# Predefined responses for off-topic questions
def handle_off_topic_questions(user_input):
    irrelevant_queries = ["how are you", "what are you doing", "what is your name"]
    if user_input.lower() in irrelevant_queries:
        return ("I'm your university information assistant, here to help with UOE-related queries. "
                "Please ask me about university topics such as admissions, scholarships, exams, and more.")

# Maintain conversation context
def maintain_conversation_context(user_input):
    conversation_history.append(user_input)

# Detect language (Roman Urdu or English)
def detect_language(user_input):
    lang, _ = langid.classify(user_input)
    return lang

# Clean the input before spelling correction
def clean_input(user_input):
    return re.sub(r'[^a-zA-Z0-9\s]', '', user_input)

# Function to replace specific terms for better matching
def replace_specific_terms(user_input):
    replacements = {
        "bscs": "bscs computer science",
        "cs": "bscs computer science", 
        "bs cs": "bscs computer science",
        "bsit": "bsit information technology",
        "bs it": "bsit information technology",
        "it": "bs information technology",
    }
    words = user_input.lower().split()
    corrected_words = [replacements.get(word, word) for word in words]
    return ' '.join(corrected_words)

# Correct spelling mistakes and handle None values
def correct_spelling(user_input):
    user_input = clean_input(user_input.lower().strip())  # Clean input
    words = user_input.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)

# Match user input with dataset using similarity
def match_intent(user_input, threshold_low=0.5, threshold_high=0.7):
    corrected_input = correct_spelling(user_input)
    user_input_embedding = similarity_model.encode(corrected_input.lower().strip(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_input_embedding, dataset_embeddings)

    best_match_idx = torch.argmax(similarities).item()
    best_similarity_score = similarities[0][best_match_idx].item()

    if best_similarity_score > threshold_high:
        return dataset_answers[best_match_idx]
    elif best_similarity_score > threshold_low:
        return dataset_answers[best_match_idx]
    
    return None  # Return None when no good match is found

# Handle single-word or keyword-based input
def handle_keyword_input(user_input):
    user_input = user_input.lower().strip()
    for item in dataset:
        if user_input in item['input'].lower():
            return item['response']
    return None

# Generate fallback response
def generate_fallback_response():
    return ("Could you please rephrase? "
            "I specialize in UOE-related information. Please ask about university topics like admissions, scholarships, exams, etc.")

# Main chatbot response function
def chatbot_response(user_input):
    user_input = normalize_spacing(user_input)
    user_input = replace_specific_terms(user_input)  # Apply specific term replacement

    maintain_conversation_context(user_input)

    lang = detect_language(user_input)
    if lang == 'ur':
        return "It seems you're asking in Roman Urdu. ask in English?"

    greeting_response = handle_greeting(user_input)
    if greeting_response:
        return greeting_response

    off_topic_response = handle_off_topic_questions(user_input)
    if off_topic_response:
        return off_topic_response

    keyword_response = handle_keyword_input(user_input)
    if keyword_response:
        return keyword_response

    matched_response = match_intent(user_input)
    if matched_response:
        if user_input in answered_questions:
            return f"I've already answered that: {answered_questions[user_input]}"
        answered_questions[user_input] = matched_response
        return matched_response

    return generate_fallback_response()  # Ensure chatbot doesn't crash when no intent is matched

# Example interaction loop
if __name__ == "__main__":
    print("Chatbot is starting...")
    print("Hello! 🎓 Welcome to **Academic Navigator** – the university information assistant for UOE.")
    print("I have the answers to your university-related questions, such as admissions, scholarships, exams, and more.")
    print("How can I assist you today? Type your question below or 'exit' to quit.")

    while True:
        user_input = input("You: ").strip().lower()
        if user_input in {"exit", "bye", "quit"}:
            print("Chatbot: Thank you for chatting! Have a great day. 😊")
            break

        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")



