from Model.actions import buildModel, trainModel, evaluateModel, saveModel, loadModel

class Model:
	def __init__(self, categories, dataSize=100000000, evaluationSize=100000000, resolution=(100,100), colour=False, name="myModel.model", ):
		self.model = None

		self.categories = categories
		self.dataSize = dataSize
		self.evaluationSize = evaluationSize
		self.resolution = resolution
		self.colour = colour
		self.name = name


	def build(self, optimizer='adam'):
		shape = [*self.resolution]
		if self.colour:
			shape.append(3)

		self.model = buildModel(len(self.categories), shape, optimizer)

	def train(self, dataSize=None, epochs=5,):
		if not dataSize:
			dataSize = self.dataSize
			
		trainModel(self.model, self.categories, self.colour, self.resolution, dataSize, epochs)

	def evaluate(self, evaluationSize=None, showImage=False):
		if not evaluationSize:
			evaluationSize = self.evaluationSize
		evaluateModel(self.model, self.categories, self.colour, self.resolution, evaluationSize, showImage)

	def save(self, name=None):
		saveModel(self.model, name if name else self.name)

	def load(self, name=None):
		loadModel(name if name else self.name)