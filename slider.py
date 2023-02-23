import imutils
import math
from featuresourcer import HogFeatureExtractor, CannyFeatureExtractor


class Slider:
  
  def __init__(self, sourcer, classifier, frame, increment):
    self.sourcer = sourcer
    self.classifier = classifier
    self.i = increment
    self.h = sourcer.h
    self.w = sourcer.w
    self.frame = frame
    sourcer.new_frame(self.frame)

  """ Function that implements the resizing loop for the input image given a certain scale."""
  def pyramid(self,image, scale=1.5, minSize=(64, 48)):
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

  """ Function for moving the sliding window"""
  def sliding_window(self,image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
      for x in range(0, image.shape[1], step_size):
        # yield the current window
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


  """ Function for the detection of garbage in the test images. It iterates the image at different scales and positions.
      Returns in output the bounding boxes associated with the detected areas"""
  def locate(self, image):
    boxes = []
    w_w,w_h = self.w,self.h
    # loop over the image pyramid
    scale = 1.5
    iteration = 0
    step_size = self.i
    for resized in self.pyramid(image=image, scale=scale):
      if (int(resized.shape[1]/w_w) <= 3 and int(resized.shape[0]/w_h) <= 3):
        # loop over the sliding window for each layer of the pyramid
        step_size = max(1,int(32/(pow(scale,iteration))))
        for (x, y, window) in self.sliding_window(resized, step_size=step_size, window_size=(w_w, w_h)):
          # if the window does not meet the desired window size, ignore it
          if window.shape[0] != w_h or window.shape[1] != w_w:
            continue

          if(type(self.sourcer) is HogFeatureExtractor):
            features = self.sourcer.slice(x, y, w_w, w_h) # get hog
          elif(type(self.sourcer) is CannyFeatureExtractor):
            features = self.sourcer.slice() # get canny
          if self.classifier.predict(features):
            boxes.append((int(x*math.pow(scale,iteration)), int(y*math.pow(scale,iteration)), (int(w_w*math.pow(scale,iteration)),int(w_h*math.pow(scale,iteration)))))


      iteration = iteration + 1

    return boxes
