import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model  # <- para cargar el modelo ya entrenado

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('punkt_tab')  # Nuevo
nltk.download('wordnet')

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar archivo intents.json
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# 🔹 Aquí ya NO volvemos a construir words y classes desde cero,
# sino que usamos las que guardó train.py:
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# 🔹 Cargar el modelo entrenado
model = load_model('chatbot_model.h5')


# --- Funciones ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):  
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    for i in intents_json['intents']:
        if i["tag"] == tag:
            return random.choice(i['responses'])

def respuesta(message):  
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res


print("Bot listo! Escribe algo para hablar con él (o 'salir' para terminar)")

while True:
    user_input = input("Tú: ")
    if user_input.lower() in ["salir", "exit", "adiós"]:
        print("Bot: ¡Hasta luego!")
        break

    response = respuesta(user_input)
    print("Bot:", response)
