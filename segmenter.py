import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

class WatershedSegmenter:
    def __init__(self, image, object_proposals):
        self.image = image
        self.object_proposals = object_proposals

    def segment_object_proposals(self):
        segmented_regions = []
        for box in self.object_proposals:
            x_min,x_max = box[0][0],box[1][0]
            y_min,y_max = box[0][1],box[1][1]
            cropped_region = self.image[y_min:y_max, x_min:x_max]

            cropped_region = cv2.cvtColor(cropped_region, cv2.COLOR_HLS2RGB)
            gray = cv2.cvtColor(cropped_region,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
            # sure background area

            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            show_img = cv2.resize(sure_bg, (500,350))
            cv2.imshow("prova",show_img)
            cv2.waitKey(0)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
            show_img = cv2.resize(sure_fg, (500,350))
            cv2.imshow("prova",show_img)
            cv2.waitKey(0)
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            markers = cv2.watershed(cropped_region,markers)
            cropped_region[markers == -1] = [255,0,0]
            show_img = cv2.resize(cropped_region, (500,350))
            cv2.imshow("prova",show_img)
            cv2.waitKey(0)
            segmented_regions.append(cropped_region)
        return segmented_regions

    def build_final_image(self, segmented_regions):
        final_image = self.image.copy()

        for i,region in enumerate(segmented_regions):
            x_min,x_max = self.object_proposals[i][0][0],self.object_proposals[i][1][0]
            y_min,y_max = self.object_proposals[i][0][1],self.object_proposals[i][1][1]
            
            final_image[y_min:y_min+region.shape[0], x_min:x_min+region.shape[1]] = region
        return final_image