**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/afterpreprocess.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_web_pics/1.png "Traffic Sign 1"
[image5]: ./test_web_pics/2.png "Traffic Sign 2"
[image6]: ./test_web_pics/3.png "Traffic Sign 3"
[image7]: ./test_web_pics/4.png "Traffic Sign 4"
[image8]: ./test_web_pics/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lifeunion/gtsclassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set. 

I used the numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I can change the contrast of the image as black and white pixels. This helps identifying especially when the lighting of the images are too dark. Besides grayscaling, normalizing is also applied to ease the classifier' job later on. They definitely have lighter burden with the values constrained to 80% on a bell curve.

Here is an example of a traffic sign image after grayscaling and normalizing.

![alt text][image2]

####2. Model architecture

My model consisted of the following layers, following LeNet basically:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120        							|
| RELU					|												|
| Fully connected		| Outputs 84        							|
| RELU					|												|
| Fully connected		| Outputs 43									|
| Softmax       		| 												|

####3. Model training

To train the model, I used learning rate of 0.0015 and batch size of 90. By 250 epochs I notice most of the times the accuracy plateaud already.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.94 
* test set accuracy of 0.926

Approach chosen to come up with this architecture is to choose starting from LeNet architecture instead of iteratively finding one. The initial validation accuracy is 89%. After applying grayscaling, normalization on the images during pre-process stage, the accuracy has already reached 93%.
* The reason I  believed LeNet architecture to be relevant to the traffic sign application is because it has the building blocks for all convolutional neural networks: convolution unit, ReLUs, pooling and fully connected layer. It is however pretty shallow, but for this task of achieving 93% accuracy, which is pretty low, it is sufficient.
* The model's accuracy on the training, validation and test set provide evidence that the model is working well. This is shown especially by close numbers of validation and test accuracies. 
 

###Test a Model on New Images

####1.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth image might be difficult to classify because the outer circle is common accross the speed limit signs but this one (no vehicle sign) is empty in the inner circle.

####2. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h speed limit  	| 30 km/h speed limit        					| 
| Speed bumps     		| Speed bumps 									|
| Straight ahead		| Straight ahead   								|
| No vehicles	      	| 30 km/h speed limit 			 				|
| Go straight or left	| Go straight or left      						|
| Caution            	| Caution 		      							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 83.3%

####3. Top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all but fourth one out of all 6 images, the model is relatively sure to infer and classify the images. 1st,2nd,3rd,5th images all result in 100% surety. 6th image has 99% surety and 1% predicting traffic signal instead of caution. This shows that the weakness of this network is to identify the inner writing. The fourth image (no vehicle sign) has the following probability.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| 30 km/h speed limit   						| 
| .49     				| No vehicle 									|
| .00					| Priority road									|
| .00	      			| End of no passing				 				|
| .00				    | Roundabout mandatory  						|


