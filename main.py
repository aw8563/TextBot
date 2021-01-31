#!/usr/bin/python3

from Model.model import Model

def main():
	# List of categories to sort into
	categories = [
		"male",
		# "female",
		"cat",
		# "dog",
	]

	model = Model(categories, dataSize=1000, evaluationSize=1000)		
	model.build()
	model.train()

	model.evaluate()
	model.evaluate(10, True)

if __name__ == "__main__":
	main()