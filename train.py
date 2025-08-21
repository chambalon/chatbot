import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
  data = json.load(file)

corpus = []
labels = []
classes = []

# Preprocessing
for intent in data['intents']:
  for pattern in intent['patterns']:
    tokens = word_tokenize(pattern)
    lemmas = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    corpus.append(" ".join(lemmas))
    labels.append(intent['tag'])
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

# Text classification
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus).toarray()
y = np.array(labels)

# Model training
model = MultinomialNB()
model.fit(x, y)

with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
  pickle.dump(vectorizer, f)
with open('classes.pkl', 'wb') as f:
  pickle.dump(classes, f)




