# 參考資料 https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions
import time
import numpy as np
import cv2
import skimage.feature
import os
from tempfile import TemporaryFile


r = 0.48     #scale down
width = 100 #patch size
#img_path = "../corpus/KaggleNOAASeaLions"		# 663 image
img_path = "/home/ray/SeaLion/Train_full"	# 20  image
#img_path = "../corpus/KaggleNOAASeaLions_mini"		# 2   image

def GetData(filename):
    # read the Train and Train Dotted images
    image_1 = cv2.imread(img_path + "/TrainDotted/" + filename)
    image_2 = cv2.imread(img_path + "/Train/" + filename)    
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

# get file list
#filelist = os.listdir("/home/alvin/cei/notebook_home/ai_lab_image/corpus/KaggleNOAASeaLions/Train/")
filelist = os.listdir(img_path + "/Train/")
filelist = filter(lambda x: x.endswith('.jpg'), filelist)
filelist = list(filelist)
filelist.sort(key=lambda x : int(x[:len(x)-4]))
filelist = list(filelist)
print('Total traing file = ' + str(len(filelist)))



#prepare training vector

final_trainX = np.array([])
final_trainY = np.array([])
final_testX  = np.array([])
final_testY  = np.array([])

#n_train = int(len(filelist)*0.7)
n_train = int(len(filelist))
print('loading training data ...')

trainset = np.arange(n_train)
for i in trainset:
    
    trainX, trainY,flag = GetData(filelist[i])
    if(flag == 'ok'):
        print('Training '+ str(i) + '\t' + str(filelist[i]))
    else:
        print('Training '+ str(i) + '\t' + str(filelist[i]) + '\t' + flag)
        continue
    
    np.random.seed(1004)
    randomize = np.arange(len(trainX))
    np.random.shuffle(randomize)
    trainX = trainX[randomize]
    trainY = trainY[randomize]
    
    final_trainX = np.vstack([final_trainX, trainX]) if final_trainX.size else trainX
    final_trainY = np.vstack([final_trainY, trainY]) if final_trainY.size else trainY

np.save('o_trainx', final_trainX)
np.save('o_trainy', final_trainY)
