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

            #cropped_region = cv2.cvtColor(cropped_region, cv2.COLOR_HLS2RGB)
            # Create a kernel that we will use to sharpen our image
            # an approximation of second derivative, a quite strong kernel
            #kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
            # do the laplacian filtering as it is
            # well, we need to convert everything in something more deeper then CV_8U
            # because the kernel has some negative values,
            # and we can expect in general to have a Laplacian image with negative values
            # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
            # so the possible negative number will be truncated
            #imgLaplacian = cv2.filter2D(cropped_region, cv2.CV_32F, kernel)
            #sharp = np.float32(cropped_region)
            #imgResult = sharp - imgLaplacian
            # convert back to 8bits gray scale
            #imgResult = np.clip(imgResult, 0, 255)
            #imgResult = imgResult.astype('uint8')
            #imgLaplacian = np.clip(imgLaplacian, 0, 255)
            #imgLaplacian = np.uint8(imgLaplacian)
            #cv.imshow('Laplace Filtered Image', imgLaplacian)
            #cv2.imshow('New Sharped Image', imgResult)
            #show_img = cv2.resize(imgResult, (int(350/0.75),350))
            #cv2.imshow("New Sharped Image",show_img)
            #cv2.waitKey(0)

            # Create binary image from source image
            try:
                bw = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                #bw = cv2.adaptiveThreshold(bw,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
                """show_img = cv2.resize(bw, (350,int(350/0.5)))
                cv2.imshow('Binary Image', show_img)
                cv2.waitKey(0)"""
                # Perform the distance transform algorithm
                kernel1 = np.ones((5,5), dtype=np.uint8)
                bw = cv2.dilate(bw, kernel1)
                dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
                # Normalize the distance image for range = {0.0, 1.0}
                # so we can visualize and threshold it
                cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                """show_img = cv2.resize(dist, (350,int(350/0.5)))
                cv2.imshow('Distance Transform Image', show_img)
                cv2.waitKey(0)"""


                # Threshold to obtain the peaks
                # This will be the markers for the foreground objects
                _, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)
                # Dilate a bit the dist image
                kernel1 = np.ones((5,5), dtype=np.uint8)
                dist = cv2.dilate(dist, kernel1)
                """show_img = cv2.resize(dist, (350,int(350/0.5)))
                cv2.imshow('Peaks', show_img)
                cv2.waitKey(0)"""
                # Create the CV_8U version of the distance image
                # It is needed for findContours()
                dist_8u = dist.astype('uint8')
                # Find total markers
                contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Create the marker image for the watershed algorithm
                markers = np.zeros(dist.shape, dtype=np.int32)
                # Draw the foreground markers
                for i in range(len(contours)):
                    cv2.drawContours(markers, contours, i, (i+1), -1)
                # Draw the background marker
                cv2.circle(markers, (5,5), 3, (255,255,255), -1)
                markers_8u = (markers * 10).astype('uint8')
                """show_img = cv2.resize(markers_8u, (350,int(350/0.5)))
                cv2.imshow('Markers', show_img)
                cv2.waitKey(0)"""
                # Perform the watershed algorithm
                cv2.watershed(cropped_region, markers)
                #mark = np.zeros(markers.shape, dtype=np.uint8)
                mark = markers.astype('uint8')
                mark = cv2.bitwise_not(mark)
                # uncomment this if you want to see how the mark
                # image looks like at that point
                #cv.imshow('Markers_v2', mark)
                # Generate random colors
                colors = []
                for contour in contours:
                    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
                # Create the result image
                dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
                # Fill labeled objects with random colors
                for i in range(markers.shape[0]):
                    for j in range(markers.shape[1]):
                        index = markers[i,j]
                        if index > 0 and index <= len(contours):
                            dst[i,j,:] = colors[index-1]
                # Visualize the final image
                """show_img = cv2.resize(dst, (350,int(350/0.5)))
                cv2.imshow('Final Result', show_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()"""

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
            print(" REGION SHAPE ",region.shape)
            try:
                final_image[y_min:y_max, x_min:x_max] = final_image[y_min:y_max, x_min:x_max]*alpha + region*(1-alpha)
            except:
                pass
        return final_image