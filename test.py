import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json

lemmatizer = WordNetLemmatizer()

# unique = set()

# with open('data.json') as f:
# 	for line in f:
# 		data = json.loads(line)
# 		text = data['reviewText']
		
# 		words = word_tokenize(text.lower())

# 		lemmatizedWords = []
# 		for word in words:
# 			word = lemmatizer.lemmatize(word)		
# 			unique.add(word) 



# with open('eval.json') as f:
# 	for line in f:
# 		data = json.loads(line)
# 		text = data['reviewText']
		
# 		words = word_tokenize(text.lower())

# 		lemmatizedWords = []
# 		for word in words:
# 			word = lemmatizer.lemmatize(word)		
# 			unique.add(word) 

# print(len(unique))


unique = set()
m = 0
with open('train.json') as f:
	for line in f:
		try:
			data = json.loads(line)
		except:
			print(line)


		text = data['text']
		
		words = word_tokenize(text.lower())

		for word in words:
			word = lemmatizer.lemmatize(word)		
			unique.add(word) 

		m = max(m, len(words))
print(len(unique))