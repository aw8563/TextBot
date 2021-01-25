import os
import random

import numpy as np
from Processing.image import processImage

# processes training data from images
# count specifies maximum images to process
# if randomise is true, randomly flip image for better training data
def processData(folders, labels, colour, resolution, count=100000000, randomise=False):
	dataSet = []
	for folder, label in zip(folders, labels):
		print("Parsing", folder, "...", end=" ")
		n = 0
		for file in getFilesInFolder(folder, count):

			try:
				n += 1

				arr, img = processImage(file, colour, resolution, randomise=randomise)
				dataSet.append((arr, label, img, file))
			except:
				# skip non images
				pass
		print(n)
	
	random.shuffle(dataSet)

	data = np.array([data[0] for data in dataSet])
	labels = np.array([data[1] for data in dataSet])
	images = [data[2] for data in dataSet]
	files = [data[3] for data in dataSet]

	return data, labels, images, files


# gets all files in folder
def getFilesInFolder(folder, count):
	if folder[-1] != "/":
		folder += "/"

	files = [folder + file for file in os.listdir(folder)]
	random.shuffle(files)

	return files if count > len(files) else files[:count]