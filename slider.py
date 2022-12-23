import numpy as np 
import imutils
import cv2
import time
from featuresourcer import FeatureSourcer
from binaryclassifier import BinaryClassifier

class Slider:
  
  def __init__(self, sourcer, classifier, increment):
    self.sourcer = sourcer
    self.classifier = classifier
    self.i = increment
    self.h = sourcer.h
    self.w = sourcer.w
    self.current_strip = None 
    
  def prepare(self, frame, wp, ws):
    x,y = wp
    w_w,w_h = ws
    print(frame.shape,"w_h ",w_h," self.h ",self.h)
    scaler = w_h / self. h
    y_end = int(y + w_h) # ending y position of the window
    x_end = int(x + w_w)
    #w = np.int(frame.shape[1] / scaler)
    w = self.w
    
    strip = frame[y: y_end, x:x_end, :] #defining the portion of image corresponding to the window
    strip = cv2.resize(strip, (w, self.h)) # resizing the window to the size of the training images in order to have the same number of features
    self.current_strip = strip 
    
    return scaler, strip

  def strip(self):
    return self.current_strip


  def pyramid(self,image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
      # compute the new dimensions of the image and resize it
      w = int(image.shape[1] / scale)
      image = imutils.resize(image, width=w)
      # if the resized image does not meet the supplied minimum
      # size, then stop constructing the pyramid
      if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        break
      # yield the next image in the pyramid
      yield image

  """def locate(self, frame, window_size, window_position):
    print("WINDOW SIZE ",window_size)
    w_w,w_h = window_size
    x,y = window_position
    scaler, strip = self.prepare(frame, window_position, window_size)
    
    boxes = []
    self.sourcer.new_frame(strip)
    
    x_end = (strip.shape[1] // self.w - 1) * self.w
    y_end = (strip.shape[0] // self.h - 1) * self.h
    print("strip shape ",strip.shape," i ",self.i)
    print("X_END ",x_end, "Y_END ",y_end)
    for resized_y in range(0, x_end, self.i):
      for resized_x in range(0, y_end, self.i):
        print("scorro")
        features = self.sourcer.slice(resized_x, resized_y, self.w, self.h) # get hog 
        print(self.classifier.predict(features))
        if self.classifier.predict(features): 
          
          x = np.int(scaler * resized_x)
          boxes.append((x, y, window_size))
        
    return boxes"""

      
  def sliding_window(self,image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
      for x in range(0, image.shape[1], stepSize):
        # yield the current window
        yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


  def locate(self, image, window_size, window_position):
    boxes = []
    w_w,w_h = window_size
    # loop over the image pyramid
    for resized in self.pyramid(image=image, scale=1.5):
      # loop over the sliding window for each layer of the pyramid
      for (x, y, window) in self.sliding_window(resized, stepSize=32, windowSize=(w_w, w_h)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != w_h or window.shape[1] != w_w:
          continue
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW
        # since we do not have a classifier, we'll just draw the window
        print("X ",x," Y ",y, " W_W ",w_w," W_H ",w_h, "WINDOW SHAPE ",window.shape)
        features = self.sourcer.slice(x, y, w_w, w_h) # get hog 
        print(" FEATURE SHAPE ", features.shape)
        #print(self.classifier.predict(features))
        #if self.classifier.predict(features): 
        #  boxes.append((x, y, window_size))
        
  
        #clone = resized.copy()
        #cv2.rectangle(clone, (x, y), (x + w_w, y + w_h), (0, 255, 0), 2)
        #cv2.imshow("Window", clone)
        #cv2.waitKey(1)
        #time.sleep(0.025)
    return boxes
