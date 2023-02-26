# svm-garbage-detection-and-segmentation
Project for the course **Signal, Image and Video** at University of Trento 2022/2023
---
The project's goal is to develop a system for the detection and segmentation of various kinds of litter in enviromental pictures.  
Since the course was focused on feature extraction and image/video processing techniques, the project attempts a more traditional approach to tackle the problem.          
  It consists of three modules:
- Training
- Garbage Detection
- Garbage Segmentation

---
## Training
A SVM has been trained separately after extracting HOG features and features from Canny contours obtained from pictures of images. This has been trained on images representing trash in a controlled enviroment (large and on a white background).
## Garbage Detection
The trained models have been then applied to images containing garbage in the environment. A sliding window has been applied to the image pyramid and the detected trash has been marked with a bounding box.
## Garbage Segmentation
The watershed algorithm has been applied to the content of bounding boxes in order to
## Demos
Demos of the project can be found [here](https://github.com/momentino/GarbageDetectionAndSegmentation/tree/main/notebooks/demo)  
Models already trained have been provided [here](https://github.com/momentino/GarbageDetectionAndSegmentation/tree/main/saved_models).  
## Results
Although the results are not completely perfect, this project allowed me to learn more about feature extraction techniques such as HOG, image augmentation, the watershed algorithm, and generally about creating a system for the detection of objects in larger images.

