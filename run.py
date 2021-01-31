#!/usr/bin/python3
import json
import random

import numpy as np
from matplotlib import pyplot

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten,Dropout,Embedding,LSTM
from tensorflow.keras.optimizers import Adam

MAX_LEN = 53
N_UNQIUE = 15075

def check(idx):
	out = np.argmax(result[idx])
	exp = evalLabels[idx]

	print("========================================================")
	print(result[i])
	print("Prediction:", out)
	print("Actual:", exp)
	print(text[idx])
	print("========================================================")

	return abs(out - exp)

def parse(file):
	lemmatizer = WordNetLemmatizer()
	with open(file) as f:
		with open('new.json', 'w') as f2:
			res = []

			for line in f:
				data = json.loads(line)
				text = data['text']
				label = int(data['score']) 
				
				words = word_tokenize(text.lower())

				lemmatizedWords = []
				for word in words:
					word = lemmatizer.lemmatize(word)
				
					unique.add(word) 
					lemmatizedWords.append(word)

				res.append((lemmatizedWords, label, text))


	random.shuffle(res)

	data = [data[0] for data in res]
	labels = [data[1] for data in res]
	text = [data[2] for data in res]

	tokenizer = Tokenizer(num_words=N_UNQIUE) # number of unique words (for both training and eval)
	tokenizer.fit_on_texts(data)
	data = tokenizer.texts_to_sequences(data) # process into array of ints
	data = sequence.pad_sequences(data, maxlen=MAX_LEN) # pad the ending with zeros if the review is not MAX_LEN words long

	return data, labels, text


try:
	model = load_model("MovieReview.model")
except:
	data, labels, _ = parse('train.json')
	# build model
	model = Sequential()
	model.add(Embedding(N_UNQIUE, 300, input_length=MAX_LEN))
	model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
	model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
	model.add(Dense(100,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(5,activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])

	model.fit(data, labels, epochs=10)
	model.save('MovieReview.model')		

evalData, evalLabels, text = parse('eval.json')

_, accuracy = model.evaluate(evalData, evalLabels, verbose=2)
print(accuracy)

result = model.predict(evalData)
diff = 0
for i in range(len(result)):
	diff += check(i)

print('difference of', diff, "over", len(result), "predctions")