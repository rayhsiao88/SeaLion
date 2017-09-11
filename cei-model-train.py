
#%matplotlib inline
import time
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import skimage.feature
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model, Model
from tempfile import TemporaryFile

#img_path = "../corpus/KaggleNOAASeaLions"
#img_path = "../corpus/KaggleNOAASeaLions_small"
#img_path = "../corpus/KaggleNOAASeaLions_mini"

T_Model = "VGG16" #VGG16/VGG19/SIMPLE
outfile= open("result_file", "a+")
outfile.write(time.strftime("%Y %b %d %H:%M:%S", time.gmtime())+"\n")

final_trainX = np.load('o_trainx_r48_100x_full_flip_PN')
final_trainY = np.load('o_trainy_r48_100x_full_flip_PN')

Model_Save_File_Name='vgg16_r48_100x_full_flip_PN.h5'

print("Using Model: "+ T_Model)
outfile.write("Using Model: "+ T_Model + '\n')

Total_start_time= time.time()

r = 0.48     #scale down
width = 100 #patch size




if(T_Model == "SIMPLE"):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(width,width,3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(5, activation='linear'))
elif(T_Model == "VGG16"):
    initial_model = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(width, width,3))
    last = initial_model.output
    x = Flatten()(last)
    x = Dense(1024)(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=.1)(x)
    preds = Dense(5, activation='linear')(x)
    model = Model(initial_model.input, preds)
elif(T_MODEL == "VGG19"):
    initial_model = keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=    (width, width,3))
    last = initial_model.output
    x = Flatten()(last)
    x = Dense(1024)(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=.1)(x)
    preds = Dense(5, activation='linear')(x)
    model = Model(initial_model.input, preds)
#else:
    #need error handling

start_time = time.time()
optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
model.compile(loss='mean_squared_error', optimizer=optim)
hist = model.fit(final_trainX, final_trainY, epochs=8, verbose=2)
print(str(hist.history))
outfile.write(str(hist.history)+"\n")
end_time = time.time()
print('\nspend: ' + str(end_time-start_time)+'s')
outfile.write('\nspend ' + str(round(end_time-start_time,1))+'s\n')


start_time = time.time()
optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=optim)
hist = model.fit(final_trainX, final_trainY, epochs=30, verbose=2)
print(str(hist.history))
outfile.write(str(hist.history)+"\n")
end_time = time.time()
print('\nspend: ' + str(end_time-start_time)+'s')
outfile.write('\nspend ' + str(round(end_time-start_time,1))+'s\n')

model.save('cei-sea-lion-model.h5')
model.save(Model_Save_File_Name)

Total_end_time= time.time()
Total_time = Total_end_time - Total_start_time
print('\nTotal spend: ' + str(Total_time)+'s')
outfile.write('\nspend ' + str(round(Total_time,1))+'s\n')




