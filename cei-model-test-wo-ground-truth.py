import re
import time
import numpy as np
import cv2
import skimage.feature
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from tempfile import TemporaryFile

r = 0.48     #scale down
width = 100 #patch size 
Tbeg_time = time.time()

#img_path = "../corpus/KaggleNOAASeaLions"
img_path = "/home/ray/SeaLion/Train_full/Test/"
#img_path = "../corpus/KaggleNOAASeaLions_mini"

model = load_model('vgg16_r48_100x_full_flip_PN.h5')
f = open('vgg16-30-r48-100-100x_flip_PN.csv', 'w')


def GetData(filename):
    # read the Train and Train Dotted images
    image_2 = cv2.imread(img_path + filename)    
    h,w,d = image_2.shape
    img = cv2.resize(image_2, (int(w*r),int(h*r)))
    h1,w1,d = img.shape

    trainX = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])

    return np.array(trainX) 
    #return trainX,trainY

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


filelist = os.listdir(img_path)
filelist = filter(lambda x: x.endswith('.jpg'), filelist)
filelist = list(filelist)
filelist.sort(key=lambda x : int(x[:len(x)-4]))
filelist = list(filelist)
print('Total tesing file = ' + str(len(filelist)))
 


n_train      = len(filelist)
trainset     = np.arange(n_train)
five_rmse_vec= np.array([])
dist_vec     = np.zeros((1,5))

empty_vec    = np.zeros(5).astype('int')
f.write("test_id,adult_males,subadult_males,adult_females,juveniles,pups\n")

for i in trainset:

    print('Processing: ' + str(filelist[i]) )
    trainX = GetData(filelist[i])

    result = model.predict(trainX)
    prediction_vec   = np.sum(result*(result>0.3), axis=0).astype('int')

    fid = re.sub('\.jpg','',filelist[i])
    out_content = np.insert(prediction_vec,0,fid) 
  #  out_content = re.sub('[|]','',str(out_content))
	#print(out_content) 
   # f.write(str(out_content)+'\n')
    cat = len(out_content)
    for i in range(cat):
        f.write(str(out_content[i]))
        if (i!=cat-1): f.write(',')
    f.write('\n')

f.close()
