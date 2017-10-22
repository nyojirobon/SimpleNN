# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np

def simple_net(num_classes):
    """
    Simple NN

    # Arguments
        num_classes: Integer, number of classes.
    """
    # Define model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,))) #28*28 pixel image
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
