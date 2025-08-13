import pickle
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy
import random


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json') as file:
  data = json.load(file)


lemmatizer = WordNetLemmatizer()
def predict_intent(sentence):
  sentence = " ".join([lemmatizer.lemmatize(token.lower()) for token in word_tokenize(sentence)])
  x = vectorizer.transform([sentence]).toarray()
  predicted_intent = model.predict(x)[0]
  return predicted_intent


def get_response(intent):
  for i in data['intents']:
    if i['tag'] == intent:
      response = random.choice(i['responses'])
  return response

intent = predict_intent('Hi')
print(get_response(intent))

