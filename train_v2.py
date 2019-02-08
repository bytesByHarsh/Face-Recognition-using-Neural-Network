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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"all")

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
    
class NN:
  def __init__(self, NI, NH, NO):
    # number of nodes in layers
    self.ni = NI+1 # +1 for bias
    self.nh = NH+1
    self.no = NO
    self.nt=2
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = np.ones((self.ni,1), dtype=int)
    self.ah = np.ones((self.nh,1), dtype=int)
    self.ao = np.ones((self.no,1), dtype=int)
    self.at = [1,-1]
    
    # create node weight matrices and randomly initialize [-2,2]
    self.wi = 4*np.random.rand(self.ni, self.nh-1) -2
    self.wo = 4*np.random.rand(self.nh, self.no) -2
    self.wt = 4*np.random.rand(self.nt, self.nh) -2
    
    
    
    # create last change in weights matrices for momentum
    self.ci = 4*np.random.rand(self.ni, self.nh) -2
    self.co = 4*np.random.rand(self.nh, self.no) -2
   
    
  def runNN (self, inputs):
    #print (inputs)
    if len(inputs) != self.ni-1:
      print ('incorrect number of inputs')
    inputs = inputs.reshape(self.ni-1,1)
    self.ai[1:] = inputs
    self.z2 = np.inner(self.ai.transpose(), self.wi.transpose())
    self.ah[1:] = sigmoid(self.z2).transpose()
    self.z3 = np.inner(self.ah.transpose(), (self.wo).transpose())
    self.ao = sigmoid(self.z3)
    #print(self.ao.shape)
    #self.ao = self.ao.transpose()
      
    return self.ao
      
      
  
  def backPropagate (self, targets, N, M):
    
    output_deltas = np.zeros((1,self.no))
    error=0.0
    targets = targets.reshape(1,self.ni-1)
    #print(self.ai.shape)
    #print(targets.shape)
    
    for k in range(self.no):
      error = targets[0,k] - self.ao[0,k]
      output_deltas[0,k] =  error * dsigmoid(self.ao[0,k]) 
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[0,k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[0,k] * self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #update input weights
    temp = self.ai
    for i in range (self.ni):
      for j in range (self.nh-1):
        change = hidden_deltas[j] * temp[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * ((targets[0,k]-self.ao[0,k])**2)
    return error
  
  def lvq(self):
      print ()
      #print (self.at)
      d1= (np.sum(np.subtract(self.wt[0],self.ah)))**2
      d2= (np.sum(np.subtract(self.wt[1],self.ah)))**2
      alpha = 0.3
      if d1<d2:
          self.wt[0] += alpha*(np.subtract(self.ah,self.wt[0]))
          self.wt[1] -= alpha*(np.subtract(self.ah,self.wt[1]))
      else:
          self.wt[0] -= alpha*(np.subtract(self.ah,self.wt[0]))
          self.wt[1] += alpha*(np.subtract(self.ah,self.wt[1]))
      print (np.dot(self.wt,self.ah))
      print ()
        
  def weights(self):
    print ('Input weights:')
    for i in range(self.ni):
      print (self.wi[i])
    print
    print ('Output weights:')
    for j in range(self.nh):
      print (self.wo[j])
    print ('')
  
  def test(self, inputs,targets):
    if self.runNN(inputs)==targets:
      print("Same ")
    else:
      print ("Different")
      
  def train (self, inputs,targets, max_iterations = 10, N=0.5, M=0.1):
    #self.lvq()
    for i in range(max_iterations):
      
      self.runNN(inputs)
      error = self.backPropagate(targets, N, M)
      if i % 5 == 0:
        print ('Combined error', error)
    #self.test(inputs,targets)
    
    
def sigmoid(x):
  return (1 / (1 + np.exp(np.multiply(-1,x))))
def dsigmoi(y):
    return y*(1-y)

#def sigmoid (x):
#  return math.tanh(x)
def dsigmoid (y):
  return 1 - y**2

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)



def main ():
  print("Starting...")
  myNN = NN ( 2500, 200, 2500)
  #myNN1= NN (2576,200,2)
  #star = [1.0,-1.0]
  for root,dirs,files in os.walk(image_dir):
    print ('Root:',root)
    for file in files:
      print (file)
      if file.endswith("png") or file.endswith("Jpg") or file.endswith("jpg") or file.endswith("JPG"):
        path = os.path.join(root,file)
        #open the file in greyscale
        gray = cv2.imread(path,0)
        bw = gray.reshape(-1)
        myNN.train(bw,bw)
      new_dir = BASE_DIR + "\BPN2"  
      save(new_dir,(myNN.wi,myNN.wo,myNN.wt))  



if __name__ == "__main__":
  main()
