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
    self.nh = NH
    self.no = NO
    self.nt=2
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    self.at = [1,-1]
    
    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    self.wt = makeMatrix (self.nt,self.nh)
    
    
    # initialize node weights to random vals
    randomizeMatrix ( self.wi, -2.0, 2.0 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
    randomizeMatrix ( self.wt, -2.0, 2.0 )
    
    
    # create last change in weights matrices for momentum
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)
   
    
  def runNN (self, inputs):
    #print (inputs)
    if len(inputs) != self.ni-1:
      print ('incorrect number of inputs')
    
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      sum1 = 0.0
      for i in range(self.ni):
        sum1 +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = sigmoid (sum1)
    
    for k in range(self.no):
      sum1 = 0.0
      for j in range(self.nh):        
        sum1 +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = sigmoid (sum1)

    
      
    return self.ao
      
      
  
  def backPropagate (self, targets, N, M):
    
    output_deltas = [0.0] * self.no
    error=0.0
    for k in range(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsigmoid(self.ao[k]) 
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * ((targets[k]-self.ao[k])**2)
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
    
    
def sigmoi(x):
  return (1 / (1 + math.exp(-x)))
def dsigmoi(y):
    return y*(1-y)

def sigmoid (x):
  return math.tanh(x)
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
        #use adptive threshold for each image
        pic= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
        bw = pic.reshape(-1)
        #Adaptive threshold set the lower limit t '0' but we need '-1'
        bw1=[]
        for j in range(len(bw)):
            if bw[j]==0:
                bw1.append(-1)
            else:
                bw1.append(1)
        
        myNN.train(bw1,bw1)
      new_dir = BASE_DIR + "\BPN2"  
      save(new_dir,(myNN.wi,myNN.wo,myNN.wt))  

if __name__ == "__main__":
  main()
  print("--- %s seconds ---" % (time.time() - start_time))


