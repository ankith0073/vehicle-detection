##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG.png
[image2]: ./output_images/non_HOG.png
[image3]: ./output_images/hog-sub.jpg
[image4]: ./output_images/test_images.png
[image5]: ./output_images/labels_and_ouput.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README



###Histogram of Oriented Gradients (HOG)

####1. Histogram of Oriented Gradients (HOG) is the primary feature on which the SVM classifier is trained

The code for this step is contained in the second code cell of the IPython notebook(vechicle_detection.ipynb) 

The images of the car and non-car classes were read and the following parameters gave a good accuracy on test set with SVM classifier

orient = 9, pix_per_cell = 8, cell_per_block = 2. The snippet on how HOG looks for car and non car images were as shown below 

![alt text][image1]

![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

The main motivation behind playing with parametes of skimage.HOG() were to obtain good accuracy of classfication on the test set.

The following parameters gave an accuracy of more than 98% on test set
orient = 9, pix_per_cell = 8, cell_per_block = 2


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features 

A Support vector machine classfier was used to classify vehicle non-vehicle data. The features which was used was Histogram of oriented gradients. As a single feature was used Standard scalar was not used to make different features have same scaling!

The code snippet for SVM training and find the test accuracy can be found in code cell [15] of the ipython notebook.

###Sliding Window Search

####1. Sliding window search

Owing to limited computational capacity on my average computer, i chose to use the HOG subsampling, where i find the HOG for the whole lower half of the image and subsample the HOG features and feed it into the classfiier. A scale of 1.5 for the lower half of image starting from 400 pixel to 656 pixel is chosen.(Image courtesy Udacity)

![alt_text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As the classifier was trained on just HOG features, the same features were extracted out of region proposals from the test images and fed into classifier and below is the resulting output predictions 

![alt text][image4]
---

### Video Implementation

####1. Here is the link to my video (Including the result from lane detection and vehicle detection together)
Here's a [link to my video result](./outputproject_video_good:.mp4)


####2. Method to reduce false positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

To filter the jittery vehicle detections, The positions of vehicles were fitlered over 15 consecutive frames which reduced the jittering frames by a significant amount!(Please note the length of averaging(Low pass filter is static and needs to be dynamic to handle more practical scenarios))

Here's an example result showing the heatmap thresholding from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the boutnding boxes for all the frames:

![alt text][image5]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

