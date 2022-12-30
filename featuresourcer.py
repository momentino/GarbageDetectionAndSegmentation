from skimage.feature import hog
import numpy as np
from utils import convert 
import cv2

class HogFeatureExtractor:
  def __init__(self, p, start_frame):
    
    self.color_model = p['color_model']
    self.w = p['bounding_box_w']
    self.h = p['bounding_box_h']
    
    self.ori = p['number_of_orientations']
    self.ppc = (p['pixels_per_cell'], p['pixels_per_cell'])
    self.cpb = (p['cells_per_block'], p['cells_per_block']) 
    self.do_sqrt = p['do_transform_sqrt']

    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC_img = None, None, None
    
    self.RGB_img = start_frame
    self.new_frame(self.RGB_img)

  def hog(self, channel):
    features, hog_img = hog(channel, 
                            orientations = self.ori, 
                            pixels_per_cell = self.ppc,
                            cells_per_block = self.cpb, 
                            transform_sqrt = self.do_sqrt, 
                            visualize = True, 
                            feature_vector = False)
    return features, hog_img

  def new_frame(self, frame):
    
    self.RGB_img = frame 
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape
    
    self.hogA, self.hogA_img = self.hog(self.ABC_img[:, :, 0])
    self.hogB, self.hogB_img = self.hog(self.ABC_img[:, :, 1])
    self.hogC, self.hogC_img = self.hog(self.ABC_img[:, :, 2])
    
  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):
        
    x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)
    #print(" x_start ", x_start," x_end ",x_end," y_start ",y_start," Y_end ",y_end)
    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    
    hog = np.hstack((hogA, hogB, hogC))
    print(" HOG SHAPEEEEE ",hog.shape)
    #print(" SLICE SHAPE ", hogA.shape)
    return hog 

  def features(self, frame):
    self.new_frame(frame)
    return self.slice(0, 0, frame.shape[1], frame.shape[0])

  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):
    #print(" PIX TO HOG DI x", x_pix," y ",y_pix," h_pix ",h_pix," w_pix ",w_pix)
    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.h, self.w
    
    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end

class CannyFeatureExtractor:
  def __init__(self, p, start_frame):
    
    self.color_model = p['color_model']

    self.first_thresh = p['first_thresh']
    self.second_thresh = p['second_thresh']
    
    self.RGB_img = start_frame
    self.ABC_img = None

    self.cannyA_img,self.cannyB_img,self.cannyC_img = None, None, None
    self.cannyA_features, self.cannyB_features, self.cannyC_features = None, None, None

    self.new_frame(self.RGB_img)

  def canny(self, channel):
    #print(" ENTROOOOOOOOOOOOOOOOOO ")
    canny_img = cv2.Canny(channel,
                          self.first_thresh,
                          self.second_thresh)
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt = None
    max_cnt_area = -1

    for cnt in contours:
      area = cv2.contourArea(cnt)
      #print(" AREAAAA ",area)
      if(area > max_cnt_area):
        max_cnt = cnt
        max_cnt_area = area
    #print("MAX CNT ",max_cnt)
    if(max_cnt is None):
      print(canny_img.dtype)
      cv2.imshow('Binary Image', canny_img)
      cv2.waitKey(0)
    area = cv2.contourArea(max_cnt)
    perimeter = cv2.arcLength(max_cnt,True)
    _,_,orientation = cv2.minAreaRect(max_cnt)
    x,y,w,h = cv2.boundingRect(max_cnt)
    rect_area = w*h
    extent = float(area)/rect_area

    hull = cv2.convexHull(max_cnt)
    hull_area = cv2.contourArea(hull)
    try:
      solidity = float(area)/hull_area
    except:
      solidity = -1
    equi_diameter = np.sqrt(4*area/np.pi)

    mask = np.zeros(canny_img.shape,np.uint8)
    cv2.drawContours(mask,[max_cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))

    features = np.array([area,hull_area, perimeter,orientation, extent, solidity, equi_diameter])
    #print(" FEATURES ",features)
    #print(" FEATURE TYPE ",features.dtype)

    
    return features, canny_img

  def new_frame(self, frame):
    
    self.RGB_img = frame 
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape



    #(B, G, R) = cv2.split( self.ABC_img)
    B = cv2.cvtColor(self.ABC_img, cv2.COLOR_BGR2GRAY)
    _, B = cv2.threshold(B, 40, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
   # G = cv2.cvtColor(self.ABC_img, cv2.COLOR_BGR2GRAY)
    #_, G = cv2.threshold(G, 40, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
   # R = cv2.cvtColor(self.ABC_img, cv2.COLOR_BGR2GRAY)
    #_, R = cv2.threshold(R, 40, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    self.cannyA_features, self.cannyA_img = self.canny(B)
    #self.cannyB_features, self.cannyB_img = self.canny(G)
    #self.cannyC_features, self.cannyC_img = self.canny(R)
    
    
  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):
        
    #x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)
    #print(" x_start ", x_start," x_end ",x_end," y_start ",y_start," Y_end ",y_end)
    #hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    #hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
   # hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()

    #features = np.stack((self.cannyA_features, self.cannyB_features, self.cannyC_features)) 
    features = self.cannyA_features
    return features

  def features(self, frame):
    self.new_frame(frame)
    return self.slice(0, 0, frame.shape[1], frame.shape[0])

  def visualize(self):
    return self.RGB_img, self.cannyA_img, self.cannyB_img, self.cannyC_img

  """def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):
    #print(" PIX TO HOG DI x", x_pix," y ",y_pix," h_pix ",h_pix," w_pix ",w_pix)
    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.h, self.w
    
    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end"""
