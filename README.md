**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[im-hog1-1]: ./images/hog/car-hls.jpg
[im-hog1-2]: ./images/hog/car-hsv.jpg
[im-hog1-3]: ./images/hog/car-ycrcb.jpg
[im-hog2-1]: ./images/hog/car2-hsv.jpg
[im-hog2-2]: ./images/hog/car2-hls.jpg
[im-hog3-1]: ./images/hog/large-hog-hls.jpg
[im-hog3-2]: ./images/hog/large-hog-hsv.jpg
[im-hog3-3]: ./images/hog/large-hog-ycrcb.jpg
[im-hog3-4]: ./images/hog/large-hog-luv.jpg
[im-hog3-5]: ./images/hog/large-hog-yuv.jpg
[im-hog4-1]: ./images/hog/non-car-ycrcb.jpg
[im-hog4-2]: ./images/hog/non-car-hls.jpg
[im-hog4-3]: ./images/hog/noncar-hsv.jpg
[im-win1]: ./images/win1.jpg
[im-win5]: ./images/win5.jpg
[im-win2]: ./images/win2.jpg
[im-win3]: ./images/win3.jpg
[im-win4]: ./images/win4.jpg
[im-win6]: ./images/win6.jpg
[im-heat]: ./images/heat.jpg
[im-heat-res]: ./images/heat-res.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
###Writeup / README

### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `lesson_functions.py`  line 5 `get_hog_features` function.
Execution of the feature extraction is done in `train_classifier()` function (iPython notebook, block 4)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes separated by color channels and with extracted gradient features.

![Car in hls color schema][im-hog1-1]

![Non Car in hls color schema][im-hog4-2]


**Color schema choice:**

I've displayed a set of images using various color schemas: `YCrCb`,`YUV`, `LUV`,`HSV`,`HLS`.
Visually comparing those helped to make a preliminary decision on what might be the best candidate.
Additionally I ran several train iteraitons with classifier to see how accuracy changes.
As a result HLS was selected with L and S channels to be used for feature extraction. I decided to drop H channel as it visually provided no good contrast and classifier did not produce any better result when using all HLS channels and when just using L, S channels. (results below)

![im-hog3-1]

![im-hog3-2]

![im-hog3-3]

![im-hog3-4]

![im-hog3-5]


##### Comparing how amount of channels influences accuracy of the classifier

**HLS(2ch)**
Relatively high accuracy
```
Using: 11 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5128
Extraction time: 75.4s
14.13 Seconds to train SVC...
Test Accuracy of SVC =  `0.9861, 0.9881, 0.9886`
```

**HLS(3ch)**
Using 3 channels does not give any additional benefit but increases extraction time
```Using: 11 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7284
Extraction time: 98.6s
3.82 Seconds to train SVC...
Test Accuracy of SVC =  `0.9881 , 0.9878`
```

**HLS(1ch)**
1 channel drops accuracy by about 2%

```
Extraction time: 47.8s
Using: 11 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 2972
12.56 Seconds to train SVC...
Test Accuracy of SVC =  `0.9562`
```

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on:
```
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = slice(1,3)
```


**Comparing amount of orientations:**

| Orientations | Accuracy | Time to extract | Amount of features |
| --- | --- | ---| --- |
| 6 | 98.49 | 64.9s | 3168 |
| 7* | 98.596 | 58.8s | 3560 |
| 8 | 98.72 | 59.8s | 3952 |
| 9 | 98.69 | 62.7s | 4344 |
| 10* | 98.55 | 69s | 4736 |
| 11 | 98.64-98.86 | 70.8s | 5128 |

\* - 7 and 10 were additionally checked. I ran 5 `fit' operations to find average accuracy.

#7 0.9838 0.9869 0.9861 0.9878 0.9852 avg=0.98596
#10 0.9875 0.9872 0.9864 0.9866 0.9838 avg=0.9863

For final used `8` - because it has nice symmetry :) (4 vertical directions and 4 diagonal if you divide circle into 8 sectors and accuracy is good)

##### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Full feature set is a combination of hog features, color bins and spatial 16x16 vector.

Data set is random shuffled with 20% of data being separated as test data.

Code is located in `train_classifier()` function below 'Training time...' comment.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I first tried a simple sliding-window approach, where feature where extracted for each window separately. This is not a very efficient way considering that most windows are overlapping upto 75%.
Later I switched to implementation where hog features are extracted once and then window scan is performed.

Search region is limited to
```
ystart = 380
ystop = 670
```
This reduces scans in regions where there are no cars

Windows are overlapping by 75%. 2 cells step when each window is 8 cells.

Scales per each window: 1, 1.2, 1.5 and 2
See `find_cars_in_frame()` function
Scales were selected empirically after experimenting with range starting at 0.8 to 2


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are examples of images with all positively detected windows (left) and labeled regions (right).
Labeled regions are result of thresholding heat map of all windows: 4 search results are combined into one heat map.

There are 2 thresholding steps overall:
* after single frame heat map. Code: `heatup_image()` function
* after smoothing over 15 frames of video. Code: there is a separate class called `FifoForHeatMapAcc`, that implements frame buffer and is used to accumulate results of detection. Its method `heatMap()` returns a combined heatmap over all frames that are currently in buffer.

![im-win1]

![im-win2]

Some detections are sometimes thresholded. And this is when smoothing over a set of video frames helps

![im-win3]

Other close frames are fine

![im-win6]

Leaving shadow region

![im-win4]

Entering shadow

![im-win5]


--------------------

### Video Implementation

#### 1. Video
Here's a [link to my video result](https://youtu.be/qGdULYmGOVg)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Filtering of false-positives is done by 2 types of thresholding described previously.

Bounding boxes are generated during 4 steps of window search per each video frame.
Later those are combined into a single heat map using `add_heat()`.
This method also checks how big is an overlap of regions. Currently it is set to 40%. If overlap is larger then weight of the region is set to 3, if not 1. This is done to increase contrast and weight of intersecting windows.

In order to reduce amount of false-positives on video method `add_heat()` creates a heat map over past 15 frames. Each frame has its weight on linear scale.
The most recent has the highest weight that is equal to amount of frames in buffer. Lowest = 1.

Frame regions are later summed up into a single heat map and thresholded based on the amount of frames in buffer.
The more frames we have in buffer - the the bigger total sum in heat map will be, so we need to apply higher threshold. It is set to 1.5 times the amount of frames in buffer.

Here are examples of frames with all boxes found on frames and final labeled regions.


** Smooth over several frames **
Here are 10 frames of the video (binary). Those are summed up and thresholded.
Below is an agregated heat map and labeled resulting image.

![im-heat]

![im-heat-res]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* There is relatively high amount of false positives produced by classifier which makes it harder to identify and filter out false-positives in final image. Ideally I'd try to use here DL classifier with several convolutional layers.
* Window region is statically limited to 380 to 680px, but in reality car may go uphill and downhill when this region can shift up or down. So a better algorithm is needed to identify optimal search region.
* Window scales were not tested properly for cars that are right in front of our car (large cars). Meaning larger windows might be needed.

