#-*- coding: utf-8 -*-
"""
Set target image to the first argument
 -image size should be 32*32 pixel
"""

from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys
import os
import pathlib

# Absolute path of result directory
result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')

# Load model
model = model_from_json(open(os.path.join(result_dir, 'model.json')).read())
# Load pre-trained weights
model.load_weights(os.path.join(result_dir, 'weight.h5'))

# Compile model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load image file and convert to numpy array
target_img = load_img(sys.argv[1], grayscale=True, target_size=(28,28))
target_array = img_to_array(target_img).reshape((-1, 784))

# Predict
predict = model.predict(target_array, batch_size=1, verbose=0)
print(str(predict))
print(str(np.argmax(predict)))
