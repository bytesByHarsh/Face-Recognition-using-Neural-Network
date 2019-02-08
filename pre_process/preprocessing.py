import numpy as np
import cv2
from PIL import Image
import pickle
import gzip
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
def main():
    count=0
    DEL_PERMISSION = input("Do You Want to Delete the original file?(Y/N)")
    for root,dirs,files in os.walk(BASE_DIR):
        print('Root:',root)
        #print(files)
        for file in files:
            if file.endswith("png") or file.endswith("Jpg") or file.endswith("jpg") or file.endswith("JPG"):
                path = os.path.join(root,file)
                gray = cv2.imread(path,0)
                #gray=cv2.GaussianBlur(gray,(5,5),0)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    gray = gray[y:y+h, x:x+w]
                gray = cv2.bilateralFilter(gray,7,75,75)
                pic= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                
                pic = cv2.resize(pic, (50,50),interpolation = cv2.INTER_AREA)
                newFileName = str(count)+'.jpg'
                
                path = os.path.join(root,newFileName)
                cv2.imwrite(path,pic)
                #cv2.imshow('image',pic)
                count+=1
                if DEL_PERMISSION =='Y':
                    os.remove(file)





if __name__=="__main__":
    print("Starting the pre-processing......")
    main()
