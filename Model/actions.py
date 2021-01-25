from numpy import argmax

from matplotlib import pyplot
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.preprocessing.image import load_img
from termcolor import colored as coloured

from Processing.data import processData

# builds a new model
def buildModel(nCategories, shape, optimizer):
	model = Sequential([
		Flatten(input_shape=shape),
	    Dense(128, activation='relu'), # random middle layers?
	   	Dense(nCategories), # number of classification categories
	    Softmax() # normalise to percentage
	])

	model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

	return model

# trains the model 
# count specifies how many images to use
def trainModel(model, categories, colour, resolution, dataSize, epochs):
	trainingDataFolders = ["./data/training/" + name + "/" for name in categories]
	trainingDataLabels = range(len(categories))

	# randomly flip some iamges
	data, labels, _, _ = processData(trainingDataFolders, trainingDataLabels, colour, resolution, dataSize, randomise=True)
	model.fit(data, labels, epochs=epochs)


# evaluates the model
# count specifies maximum images to load
# if show is set to true, pyplot will display image
def evaluateModel(model, categories, colour, resolution, evaluationSize, showImage):
	validationDataFolders = ["./data/validation/" + name + "/" for name in categories]
	validationDataLabels = range(len(categories))

	data, labels, images, files = processData(validationDataFolders, validationDataLabels, colour, resolution, evaluationSize)

	_, accuracy = model.evaluate(data, labels, verbose=2)

	result = model.predict(data)
	correct = 0
	wrong = 0

	resultsMap = {}
	for category in categories:
		# times correct, times wrong, total times guessed
		resultsMap[category] = [0, 0, 0]

	for res, label, img, file in zip(result, labels, images, files):
		choice = argmax(res)		
		isCorrect = choice == label

		if isCorrect:
			correct += 1
			resultsMap[categories[label]][0] += 1
		else:
			wrong += 1
			resultsMap[categories[label]][1] += 1

		resultsMap[categories[choice]][2] += 1

		if showImage:
			print(coloured("Prediction: %s, Actual: %s, Evaluation: %s" % (categories[choice], categories[label], res), \
				'green' if isCorrect else 'red'))

			# image
			pyplot.figure()
			pyplot.subplot(1,2,1)
			pyplot.imshow(img)
			pyplot.title(file)

			# prediction
			pyplot.subplot(1,2,2)
			bar = pyplot.bar(categories, res)
			pyplot.ylim([0,1])

			if isCorrect: # correct
				bar[choice].set_color("green")
			else:
				bar[choice].set_color("red")

			pyplot.show()

	print("\nRESULTS...")
	for key,value in resultsMap.items():
		print("%s - Times guessed = %d, correct = %d, wrong = %d, Accuracy = %f" % (key, value[2], value[0], value[1], value[0]/(value[0] + value[1])))
	print("\nTOTAL: %d correct, %d wrong. Accuracy = %f\n" % (correct, wrong, correct/(correct+wrong)))
	
	return accuracy

# saves model
def saveModel(model, name):
	model.save(name)


# loads model
def loadModel(name):
	model = None
	try:
		model = load_model(name)
	except:
		pass

	return model