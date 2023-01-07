from scipy.ndimage.measurements import label
from utils import box_boundaries
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class HeatMap:

  def __init__(self, frame, memory):
    
    self.blank = np.zeros_like(frame[:, :, 0]).astype(float)
    self.map = np.copy(self.blank)
    self.thresholded_map = None
    self.labeled_map = None
    self.samples_found = 0
    self.memory = memory
    self.history = []
    self.final_bounding_boxes = []

  def reset(self):
    self.map = np.copy(self.blank)
    self.history = []

  def do_threshold(self):
    self.thresholded_map = np.copy(self.map)
    self.thresholded_map[self.map < int(np.amax(self.map)*0.6)] = 0

        
  def get(self):
    self.do_threshold()
    self.label()
    return self.map, self.thresholded_map, self.labeled_map
      
  def remove(self, boxes):
    for box in boxes: 
      x1, y1, x2, y2 = box_boundaries(box)    
      self.map[y1: y2, x1: x2] -= 1
      
  def add(self, boxes): 
    for box in boxes: 
      x1, y1, x2, y2 = box_boundaries(box)
      self.map[y1: y2, x1: x2] += 1

  def update(self, boxes):
    
    if len(self.history) == self.memory:
      self.remove(self.history[0])
      self.history = self.history[1:]
    
    self.add(boxes)
    self.history.append(boxes)

  def label(self):
    labeled = label(self.thresholded_map)
    self.samples_found = labeled[1]
    self.labeled_map = labeled[0]

  def draw(self, frame, color = (0, 225, 0), thickness = 10):
    
    this_frame = frame.copy()
    _, _, this_map = self.get()
    candidate_bounding_boxes = []
    for n in range(1, self.samples_found + 1):
      coords =  (this_map == n).nonzero()
      xs, ys = np.array(coords[1]), np.array(coords[0])
      p1 = (np.min(xs), np.min(ys))
      p2 = (np.max(xs), np.max(ys))
      candidate_bounding_boxes.append((p1,p2))

    for i,box in enumerate(candidate_bounding_boxes):
      x1_i, y1_i = box[0]
      x2_i, y2_i = box[1]
      not_included = True
      for j,box2 in enumerate(candidate_bounding_boxes):
        if i != j:
          x1_j, y1_j = box2[0]
          x2_j, y2_j = box2[1]
          if((x1_j<x1_i) and (x2_j>x2_i) and (y1_j <y1_i) and (y2_j>y2_i)):
            not_included = False
            break
      if(not_included == True):
        self.final_bounding_boxes.append((box))
    self.final_bounding_boxes = list(dict.fromkeys(self.final_bounding_boxes))
    for box in self.final_bounding_boxes:
      p1,p2 = box[0],box[1]
      cv2.rectangle(this_frame, p1, p2, color, thickness)
    
    return this_frame

  def show(self, frame, path, tdpi = 80):
      
    mp, tmp, lmp = self.get()
    labeled_img = self.draw(frame)
    
    fig, ax = plt.subplots(1, 3, figsize = (15, 8), dpi = tdpi)
    ax = ax.ravel()

    ax[0].imshow(np.clip( mp, 0, 255), cmap = 'hot')

    ax[1].imshow(np.clip(tmp, 0, 255), cmap = 'hot')
    ax[2].imshow(labeled_img)
<<<<<<< HEAD
    #cv2.imwrite(path,labeled_img)
=======
>>>>>>> b3bb26cb64d80b768d61a4fc847b5d77373ced1a
    plt.show()

    for i in range(3):
      ax[i].axis('off')
  def get_final_bounding_boxes(self):
    return self.final_bounding_boxes
