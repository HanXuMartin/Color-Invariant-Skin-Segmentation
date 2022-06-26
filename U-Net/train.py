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
# tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------Shape Augmentation----------
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# ----------Training setup----------
batch_size = 8
train_size = len(os.listdir('./dataset/train/'+'image'))
val_size = len(os.listdir('./dataset/validation/'+'image'))
model_save_file = './checkpoints'
if not os.path.exists(model_save_file):
    os.makedirs(model_save_file)
print(train_size,val_size)
trainGene = trainGenerator(batch_size,'./dataset/train','image','mask',data_gen_args,save_to_dir = None)
validGene = validGenerator(batch_size,'./dataset/validation','image','mask',save_to_dir = None)  # Validation
model = unet()
model_checkpoint = ModelCheckpoint(model_save_file+'/unet_ECU-{epoch:02d}.hdf5', monitor='loss',verbose=1)
epochs = 10



history = model.fit_generator(
    trainGene,
    validation_data=validGene,
    validation_steps=val_size/batch_size, #tatalvalidationset/batchsize
    steps_per_epoch=train_size/batch_size, #totaltrainset/batchsize
    epochs=epochs,
    verbose=2, 
    shuffle=True,
    callbacks=[model_checkpoint])

imagename = 'Loss'
    # print(a)
y1 = history.history['loss']
y2 = history.history['val_loss']
    # print(y)
x = np.array(range(epochs))
    # print(x.shape)
plt.plot(x,y1, label = 'trainingloss')
plt.plot(x,y2, label = 'validloss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss'+'.png')
plt.show()

imagename = 'Accuracy'
    # print(a)
y1 = history.history['accuracy']
y2 = history.history['val_accuracy']
    # print(y)
x = np.array(range(epochs))
    # print(x.shape)
plt.plot(x,y1, label = 'trainingaccuracy')
plt.plot(x,y2, label = 'validaccuracy')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('Acc'+'.png')
plt.show()

