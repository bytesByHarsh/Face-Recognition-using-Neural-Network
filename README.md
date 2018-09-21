# Face-Recognition-using-Neural-Network
Recognize a face in an image using neural network

Dataset :- http://robotics.csie.ncku.edu.tw/Databases/FaceDetect_PoseEstimate.htm#Our_Database_

Dataset provided in this repository is has cropped faces in order to train. Original images can be found on the above link.
# Introduction
## Tools Used:-
 Python 3.6 <br/>
 Opencv 3.2.0 <br/>
 Numpy <br/>
 Pillow <br/>
 Pickle <br/>
 gzib <br/>

## Backpropogation
In artificial neural networks we use backpropagation to calculate a gradient that is needed in the calculation of the weights to be used in the network. This method is used to train deep neural networks i.e. networks with more than one hidden layer. It is equivalent to automatic differentiation in reverse accumulation mode. It requires the derivative of the loss function with respect to the network output to be known, which typically (but not necessarily) means that a desired target value is known. For this reason it is considered to be a supervised learning method.

## LVQ
LVQ stands for Learning Vector Quantization. It is a prototype-based supervised classification algorithm. LVQ is the supervised counterpart of vector quantization systems.


# Procedure

## Pre- Processing 
In the obtained database the image resolution is 680x480. If we take this as input then the total number of inputs will be 326,400 and we will be running it through multiple epoch which will take a lot of time and a lot computing power. Therefore to save time we decrease the resolution and take image size 50x50 and crop only the faces so that it gives us better results.

