# import the necessary libraries
import tensorflow as tf
import numpy as np
import math
from keras.layers import Input, Flatten, Dense, Convolution2D, Dropout, Activation, MaxPooling2D
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import csv
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Fix error with TF and Keras
tf.python.control_flow_ops = tf


# Define a function that loads data images
def loadImages():
    X_data = []
    f = open("Augmented_driving_log.csv")
    csv_f = csv.reader(f)
    i = 0
    for row in csv_f:
        if(i == 1):
            img= mpimg.imread(row[0])
	    # crop the image to exclude sky and car hood
            img= img[60:150,:,:]
            # Shrink the image for easier processing on CPU
            img = cv2.resize(img,None,fx=0.1,fy=0.1,interpolation = cv2.INTER_AREA)
            X_data.append(img)
        i = 1
    return X_data


# Define a function that loads the steering angles
def loadSteeringAngles():
    steering_angle = []
    driving_log = pd.read_csv('ttest.csv')
    #driving_log = pd.read_csv('driving_log.csv')
    steering_angle = driving_log.steering
    return steering_angle


# Define a function that normalizes the input images
def normalizeImages(inputImageArray):
    a = -0.5
    b = 0.5
    min_val = 0
    max_val = 255
    return a + ( ( (inputImageArray - min_val)*(b - a) )/( max_val - min_val) )

def main(_):
    # Load the images and steering angles into data lists    
    X_data = np.asarray(loadImages())
    y_data = np.asarray(loadSteeringAngles())

    # Normalize the images using min-max scaling, ouput rage = [-0.5,0.5]
    X_normalized = normalizeImages(X_data)


    # Split data into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_data, test_size = 0.3, random_state = 0)

    print("Training, Testing, and Validation data ready")

    # Image shape 
    image_shape = np.shape(X_data)[1:]

    # Build the model

    # Define input shape and input
    input_shape = image_shape
    inp = Input(shape=input_shape)
    
    # Define the model with compensation
    x = Convolution2D(16, 2, 2, border_mode='same')(inp)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Convolution2D(64, 5, 5, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1164)(x)
    x = Activation('relu')(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = Dense(50)(x)
    x = Activation('relu')(x)
    x = Dense(10)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)    

    
    model = Model(inp,x)

    print(model.summary())

    adam = Adam(lr = 0.0001)
    
    # create the model 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    # Hyper parameters
    epochs = 60
    batch_size = 64

    # train the model
    model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, validation_data=(X_test, y_test), shuffle=True)

    print('Finished Training!')

    ###Save Model and Weights###
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved Model!")

if __name__ == '__main__':
    tf.app.run()
    



