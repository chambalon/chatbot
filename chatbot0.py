import nltk
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


nlp = spacy.load('en_core_web_sm')
# Cleaning the text
def clean_text(text):
  doc = nlp(text.lower())
  lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
  return lemmas

print(clean_text("Hello! How can I help you today?"))

# Loads the model and it's corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

# Chat loop
chat_history_ids = None
def ask_question(question,chat_history_ids=None):
  input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')
  bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids
  chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
  response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

  return response, chat_history_ids