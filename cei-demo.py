import time
import numpy as np
import cv2
import skimage.feature
import keras
import os
import matplotlib.pyplot as plt
from PIL import Image 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from tempfile import TemporaryFile

r = 0.4     #scale down
width = 100 #patch size 
Tbeg_time = time.time()

#img_path = "../corpus/KaggleNOAASeaLions"
img_path = "../corpus/KaggleNOAASeaLions_small"
#img_path = "../corpus/KaggleNOAASeaLions_mini"


def GetData(filename):
    # read the Train and Train Dotted images
    image_1 = cv2.imread(img_path  + '/TrainDotted/' + filename)
    image_2 = cv2.imread(img_path + '/Train/' + filename)    
    img1 = cv2.GaussianBlur(image_1,(5,5),0)

    # absolute difference between Train and Train Dotted
    if(image_1.shape !=  image_2.shape):
        #skip due to size not equal
        return np.array([]),np.array([]),'error_no_2'
    image_3 = cv2.absdiff(image_1,image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4,axis=2)
  
    # alvin: check blog_log
    rmse = np.sqrt((image_3**2).mean())
    if(rmse > 8.0):
        #print('skip due to blob failed')
        return np.array([]),np.array([]),'error_no_1'
    
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h,w,d = image_2.shape

    res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25: # RED
            res[x1,y1,0]+=1
        elif R > 225 and b > 225 and g < 25: # MAGENTA
            res[x1,y1,1]+=1
        elif R < 75 and b < 50 and 150 < g < 200: # GREEN
            res[x1,y1,4]+=1
        elif R < 75 and  150 < b < 200 and g < 75: # BLUE
            res[x1,y1,3]+=1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1,y1,2]+=1

    ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w*r),int(h*r)))
    h1,w1,d = img.shape

    trainX = []
    trainY = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i,j,:])
            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])

    return np.array(trainX), np.array(trainY), 'ok'
    #return trainX,trainY

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

model = load_model('cei-sea-lion-model.h5')



filelist = os.listdir(img_path + "/Train/")
filelist = filter(lambda x: x.endswith('.jpg'), filelist)
filelist = list(filelist)
filelist.sort(key=lambda x : int(x[:len(x)-4]))
filelist = list(filelist)
print('Total tesing file = ' + str(len(filelist)))

sel_fid  = '1.jpg'
 
n_train      = len(filelist)
trainset     = np.arange(n_train)
five_rmse_vec= np.array([])
dist_vec     = np.zeros((1,5))
    
trainX, trainY,flag = GetData(sel_fid)
if(flag == 'ok'):
    #print('Testing '+ sel_fid)
    print('\n')
else:
    print('Testing\t' + sel_fid + '\t' + flag)
    exit()

result = model.predict(trainX)
ground_truth_vec = np.sum(trainY, axis=0)
prediction_vec   = np.sum(result*(result>0.3), axis=0).astype('int')
five_class_rmse  = rmse(prediction_vec,ground_truth_vec)
five_rmse_vec    = np.append(five_rmse_vec,five_class_rmse)
dist_vec        += np.abs(prediction_vec-ground_truth_vec)
    
    
print(sel_fid +' ==> CNN VGG16 output--')
print('    ground truth: ', ground_truth_vec)
print('  evaluate count: ', prediction_vec)
print('      difference: ', str(np.abs(prediction_vec-ground_truth_vec)))
print('            rmse: ', str(five_class_rmse))


# re-generate image
# plot ground-truth
image_2 = cv2.imread(img_path + '/Train/' + sel_fid)
imgplot = plt.imshow(image_2)
