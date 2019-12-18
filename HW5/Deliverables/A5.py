from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import numpy as np

from PIL import Image
import glob

import json
from sklearn.model_selection import train_test_split

def getVGGFeatures_old(img, layerName):
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	img = img.resize((224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	internalFeatures = model.predict(x)

	return internalFeatures

def getVGGFeatures(fileList, layerName):
	#Initial Model Setup
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	
	#Confirm number of files passed is what was expected
	rArray = []
	print ("Number of Files Passed:")
	print(len(fileList))

	for iPath in fileList:
		#Time Printing for Debug, you can comment this out if you wish
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)
		try:
			#Read Image
			img = image.load_img(iPath)
			#Update user as to which image is being processed
			print("Getting Features " +iPath)
			#Get image ready for VGG16
			img = img.resize((224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#Generate Features
			internalFeatures = model.predict(x)
			rArray.append((iPath, internalFeatures))			
		except:
			print ("Failed "+ iPath)
	return rArray

######################################## CODE ##########################################
def cropImage(image, x1, y1, x2, y2):
    bounds = (x1, y1, x2, y2)
    return image.crop(bounds) 

def standardizeImage(image, x, y):
	return image.resize((x,y))

def preProcessImages(raw):
    dataset = {}
    for filename in raw:

        data = filename.strip('uncropped/').strip('.jpg').split('-')
        bound = data[2].split(',')

        # print(data)
        # print(bound)

        img = Image.open(filename)

        crop = cropImage(img, int(bound[0]), int(bound[1]), int(bound[2]), int(bound[3]))
        resize = standardizeImage(crop, 60, 60)

        new_filename = 'processed/'+data[0]+data[1]+'.jpg'
        print(new_filename)
        resize.save(new_filename, 'JPEG')

        label = ''.join([i for i in data[0] if not i.isdigit()])
        print(label)
        dataset[new_filename] = label
    #end for
    return dataset

def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):

    #more processing
    X_raw = []
    for filename in preProcessedImages:
        img = Image.open(filename)
        gray = img.convert("L")

        pixels = np.array(gray).flatten()
        X_raw.append(pixels)
        print(pixels)

    label_dict = {'adcliffe':0, 'bracco':1, 'butler':2, 'harmon':3, 'ilpin':4, 'vartan':5}
    int_labels = []
    for l in labels:
        int_labels.append(label_dict[l])
    
    X = np.array(X_raw, dtype=np.float64)
    y = np.array(int_labels)

    #split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #split val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=1)
    
    #verify shape
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)

    #normalize
    X_train /= 255
    X_val /= 255
    X_test /= 255
    print("Train matrix shape", X_train.shape)
    print("Test matrix shape", X_test.shape)

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 6
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = keras.utils.to_categorical(y_train, n_classes)
    Y_val = keras.utils.to_categorical(y_val, n_classes)
    Y_test = keras.utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    #build network
    model = Sequential()
    model.add(Dense(6, input_shape=(3600,)))
    model.add(Activation('relu'))  
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #training
    history = model.fit(X_train, Y_train,
        batch_size=128, epochs=20,
        verbose=2,
        validation_data=(X_val, Y_val))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return history
#end


def trainFaceClassifier_VGG(extractedFeatures, labels):
	utils.raiseNotDefined()


def main():
    dataset = None
    # Process
    image_files = glob.glob('uncropped/*.jpg')
    print("Processing Images...")
    dataset = preProcessImages(image_files)
    print("done.")

    with open('processed-imgs.json', 'w') as fp:
        json.dump(dataset, fp, sort_keys=True, indent=4)

    # Train
    with open('processed-imgs.json', 'r') as fp:
        dataset = json.load(fp)

    images_processed = list(dataset.keys())
    labels = list(dataset.values())
    print("fitting model")
    model = trainFaceClassifier(images_processed, labels)




#end main

if __name__ == '__main__':
    main()
