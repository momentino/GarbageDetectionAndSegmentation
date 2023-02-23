import numpy as np
import cv2

import random as rng
rng.seed(12345)

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


            try:
                bw = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                kernel1 = np.ones((5,5), dtype=np.uint8)
                bw = cv2.dilate(bw, kernel1)
                dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
                cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)


                _, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)
                kernel1 = np.ones((5,5), dtype=np.uint8)
                dist = cv2.dilate(dist, kernel1)

                dist_8u = dist.astype('uint8')
                contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                markers = np.zeros(dist.shape, dtype=np.int32)

                for i in range(len(contours)):
                    cv2.drawContours(markers, contours, i, (i+1), -1)

                cv2.circle(markers, (5,5), 3, (255,255,255), -1)
                markers_8u = (markers * 10).astype('uint8')

                cv2.watershed(cropped_region, markers)

                mark = markers.astype('uint8')
                mark = cv2.bitwise_not(mark)

                colors = []
                for contour in contours:
                    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
                dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

                for i in range(markers.shape[0]):
                    for j in range(markers.shape[1]):
                        index = markers[i,j]
                        if index > 0 and index <= len(contours):
                            dst[i,j,:] = colors[index-1]

                segmented_regions.append(dst)
            except:
                pass
        return segmented_regions

    def build_final_image(self, segmented_regions):
        final_image = self.image.copy()
        alpha = 0.5
        for i,region in enumerate(segmented_regions):
            x_min,x_max = self.object_proposals[i][0][0],self.object_proposals[i][1][0]
            y_min,y_max = self.object_proposals[i][0][1],self.object_proposals[i][1][1]

            try:
                final_image[y_min:y_max, x_min:x_max] = final_image[y_min:y_max, x_min:x_max]*alpha + region*(1-alpha)
            except:
                pass
        return final_image