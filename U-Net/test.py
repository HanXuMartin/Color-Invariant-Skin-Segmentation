from model import *
from data import *
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil
import time



time_start=time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------Setup your test file and output file----------
test_file = 'C:/Users/xuhan/Desktop/ColorSpace_github/test_images'
model = keras.models.load_model('checkpoints/pretrained.hdf5') #load your model
output_file = './output' #your output file
test_names = os.listdir(test_file)
test_size = len(test_names)
testGene = testGenerator(test_file, test_size)

results = model.predict(
    testGene, batch_size=None, verbose=0, steps=test_size, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)

if not os.path.exists(output_file):
    os.makedirs(output_file)
    
saveResult(output_file,test_names,results)

image_names = os.listdir(test_file)

print('Binary-Size converter working')
for image_name in tqdm(image_names):

    image = cv2.imread(os.path.join(test_file,image_name))
    prediction = cv2.imread(os.path.join(output_file,image_name),0)
    size = (image.shape[1],image.shape[0])
    prediction = cv2.resize(prediction,size)
    ret, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_file,image_name), prediction)






