
import math
import random
import string
import numpy as np
import cv2
from PIL import Image
import pickle
import gzip
import os
import time
start_time = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"positive")
#to train for negative images uncomment below
#image_dir = os.path.join(BASE_DIR,"negative")
def sigmoid (x):
  return math.tanh(x)

def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = pickle.load(stream)
    stream.close()
    return model


def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    pickle.dump(model, stream)
    stream.close()

def to_check(w1,w2,b):
    for i in range(len(Y)):
        if (w1*x1[i] + w2*x2[i] + b*B[i])>=0:
            if Y[i]!=1:
                return 0
        elif (w1*x1[i] + w2*x2[i] + b*B[i])<0:
            if Y[i]!=-1:
                return 0
    return 1
w1,w2,b=0,0,0
letter=[]
oy=[]
oy1=[]
#load the latest one only
wi,wo,wt,w1,w2,b = load("BPN_new")
count=0
for root,dirs,files in os.walk(image_dir):
  print ('Root:',root)
  for file in files:
    print (file)
    if file.endswith("JPG") or file.endswith("jpg") or file.endswith("png"):
      path = os.path.join(root,file)
      gray = cv2.imread(path,0)

      pic= cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
      bw1 = pic.reshape(-1)
      bw=[]
      for j in range(len(bw1)):
        if bw1[j]==0:
          bw.append(-1)
        else:
          bw.append(1)
        
      np.append(bw,1.0)
      h=[]
      o=[]
      h1=[]
      o1=[]
      for j in range(200):
          sum0 = 0.0
          for i in range(len(bw)):
            sum0 +=( bw[i] * wi[i][j] )
          
          h.append(sigmoid(sum0))

      
      for k in range(2):
          sum1 = 0.0
          for i in range(200):
            sum1 +=( h[i] * wt[k][i] )
          o1.append(sum1)
      oy.append[o1[0]]
      oy1.append[o1[1]]

           
print (np.mean(oy),np.mean(oy1))
Y=1 #keep y=-1 when training for negatives
B=np.ones(len(oy))
for i in range(len(oy)):
    w1 += oy[i]*Y
    w2 += oy1[i]*Y
    b += B[i]*Y
new_dir = BASE_DIR + "\BPN_new"
save(new_dir,(wi,wo,wt,w1,w2,b))

print("--- %s seconds ---" % (time.time() - start_time))
