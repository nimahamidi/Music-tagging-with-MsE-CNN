'''
Code written by Nima Hamidi (2019 03 25)
All reserved ~~~~
'''

from music_tagger_cnn import MusicTaggerMS_CNN, MusicTaggerCNN, MusicTaggerMS2_CNN
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.utils import multi_gpu_model, Sequence
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def main(task, type):
	X_train = np.load('copy/X_train.npy')
	y_train = np.load('copy/y_train.npy')
	X_test = np.load('copy/X_test.npy')
	y_test = np.load('copy/y_test.npy')

	datagen_train = ImageDataGenerator(
    	featurewise_center=True,
    	featurewise_std_normalization=True,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True)

	datagen_valid = ImageDataGenerator(
    	featurewise_center=True,
    	featurewise_std_normalization=True,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True)

	datagen_train.fit(X_train)
	datagen_valid.fit(X_test)

	if type == "MS":
		model = MusicTaggerMS_CNN(weights=None, input_tensor=None, include_top=True)
	elif type == "Simple":
		model = MusicTaggerCNN(weights=None, input_tensor=None, include_top=True)
	elif type == "MS2":
		model = MusicTaggerMS2_CNN(weights=None, input_tensor=None, include_top=True)

	if task == "train":
		start = time.time()
		model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
		model.fit_generator(datagen_train.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train)/32, epochs=50, validation_data=datagen_valid.flow(X_train, y_train), validation_steps=50)

		model.save("model.h5")
	print ("trained in {} seconds". format(time.time()-start))

if __name__ == "__main__":
    main("train", "MS2")
