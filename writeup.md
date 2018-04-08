## Writeup 

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4-9 code cells of the IPython notebook.  

Please see IPython notebook for more sample images. It is very well commented

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I implemented these three to my model 

spatial_feat=True, hist_feat=True, hog_feat=True

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 10-13 code cells of the IPython notebook.  

Please see IPython notebook for more sample images. It is very well commented.

I trained a linear SVM using;

g_svc = LinearSVC()
para_C = {'C':[0.005,0.01,0.04,0.08,0.10,0.20,0.30,0.40,0.50,0.70,0.90,1.00,1.10,1.20]}
iter_para = 10
g_svc = RandomizedSearchCV(g_svc, param_distributions= para_C, n_iter= iter_para)


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 15 code cell of the IPython notebook.  

Please see IPython notebook for more sample images. It is very well commented.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  


The code for this step is contained in the 35 code cell of the IPython notebook.  

Please see IPython notebook for more sample images. It is very well commented.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I mainly work on false pozitive problem that my implementation detect non-cars as cars image so many times. Avoiding this problem I specified multi_find_cars function that draw multi rectangle for heatmap then I set my heatmap threshold to as much as high I can to avoid false positives. There is still some place for improvement but I believe this is valid for this assignment.

Thanks!

