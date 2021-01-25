import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img

# returns pyplot image and np arr for the image
# if randomise is true, randomly flip image
def processImage(file, colour, resolution, randomise=False):
	image = load_img(file, color_mode='rgb' if colour else 'grayscale')

	# flip along y axis half the time
	if randomise:
		image = ImageOps.mirror(image)

	# resize without keepign aspect ratio
	return np.array(image.resize(resolution, Image.ANTIALIAS)), image

# runs the main function
if __name__ == '__main__':
	main()