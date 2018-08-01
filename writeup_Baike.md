# **Traffic Sign Recognition** 

## Writeup

---
**Build a Neural Network to Recognize Traffic Signs**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./misc/dataSet_BarChart.JPG "Visualization"
[image2]: ./misc/lenet.png "Lenet Diagram"
[image3]: ./misc/modifiedLeNet.jpg "modified LetNet Model"
[image4]: ./misc/Addition_7_Germany_Traffic_Signs.JPG "7 Germany Traffic Signs"
[image5]: ./new_signs/sign3.jpg "Sign3"
[image6]: ./new_signs/sign4.jpg "Sign4"
[image7]: ./new_signs/sign5.jpg "Sign5"

---
### Files Submitted
* Two Ipython notebooks with code: 1) [Traffic_Sign_Classifier_Baike.ipynb](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Baike.ipynb)
                                   2) [Traffic_Sign_Classifier_Sermanet_Baike.ipynb](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Sermanet_Baike.ipynb)
* HTML output of the code:   1) [Traffic_Sign_Classifier_Baike.html](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Baike.html)
                             2) [Traffic_Sign_Classifier_Sermanet_Baike.html](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Sermanet_Baike.html)
* A markdown writeup report [writeup_Baike.md](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_Baike.md)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

| Item			         	     |  Size	    | 
|:------------------------------:|:------------:| 
| training set  	     | 34799 	    | 
| validation set     | 4410  		|
| test set				 | 12630		|
| Shape of traffic sign image	 | 32, 32, 3    |
| Number of unique labels		 | 43 	   		|

#### Visualization of the dataset.

Here is a visualization of the data set. It is a bar chart showing how many samples are included for each sign class:

![alt text][image1]

In order to visualize the German Traffic Signs dataset, Random selected images with their associated lable. Please see [Traffic_Sign_Classifier_Baike.html](https://github.com/baikeshen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Baike.html) for this visualization.

### Design and Test a Model Architecture

#### Preprocessing 

##### Greyscaling

As a first step, I sticked to the color images, the shape of first layer to the original LeNet has to be changed from (5 , 5, 1 ,6) to (5, 5, 3, 6). Compared to converting the images to grayscale, there is no so much diffrence from the final results perspective, but the computation is much heavier. So I decided to give up color image and picked up the mothed of converting the images to gray. It helps to reduce training time. 

##### Normalization

Secondly, Normalizing the data to range (-1, 1). This is done by using XX_train=(xx_train-128)/128. As suggested in the lesson, this way is nice and easy to implement. Following this method, there is not wider distribution in the data and make it easier to train using a singular learning rate.

#### Model architecture

I have adopted two models. For the first one I began by implementing the similar archtecture from the LetNet lab, with no changes since my dataset is in grayscale. It has two convolutional layers with max pooling and RELU activation, a flattening layer, and then 3 fully connected layers with REUL activation. The original Lenet neural architechture diagram is shown as below:

: ![alt text][image2]

Below is a description of the LeNet architecture for a 32x32x1 input greyscale image with 43 output labels.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. Input         		| 32x32x1 Greyscale image 						| 
| 2. Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 28x28x6 	|
| 3. RELU					| Activation function							|
| 4. Max pooling	      	| 2x2 stride, valid padding, outputs 16x16x6 	|
| 5. Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 10x10x16 	|
| 6. RELU					| Activation function							|
| 7. Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| 8. Flattening 	      	| outputs 5x5x16=400						 	|
| 9. Fully connected		| output 120   									|
| 10. RELU					| Activation function							|
| 11. Fully connected		| output 84   									|
| 12. RELU					| Activation function							|
| 13. Fully connected		| output 43   									|
| 14. Softmax				| Applied to get probabilities					|
|						|												| 



The second one I have adapted from Sermanet/Lecun traffic sign classififcation joural article. The archtecture diagram is shown as below:
: ![alt text][image3]

Based on my own understanding, the model is set up as below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. Input         		| 32x32x1 Greyscale image 						| 
| 2. Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 28x28x6 	|
| 3. RELU					| Activation function							|
| 4. Max pooling	      	| 2x2 stride, valid padding, outputs 16x16x6 	|
| 5. Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 10x10x16 	|
| 6. RELU					| Activation function							|
| 7. Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| 8. Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 1x1x400 	|
| 9. RELU					| Activation function							|
| 10. Flatten layers from 8 (1x1x400) and 6 （5x5x16 ->400) 		| output 800   									|
| 11. Concatenate flattened layers to a single size					| output 800							|
| 12. Fully connected		| output 84   									|
| 13. Fully connected layers		| output 43   									|
| 14. Softmax				| Applied to get probabilities					|
|						|												| 


#### Training

To train the model, I used Adam optimizer (already implemented in LetNet lab), the final setting used for both models is the same :
- batch size: 128
- epochs: 30
- learning rate: 0.001
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.5

It calculates loss using softmax cross entropy.  It minimizes loss using the Adam Optimizer which is built into TensorFlow.  It then checks the model on the validation set and outputs an accuracy. The final results from both model are quite similar. It is surprise to me. I thought the second one (Seramut/Lecun) should be better than the first one (original LetNet). I can not tell the reason. 

My final accuracy was: 
* 96.2% on the validation set
* 93.5% on the test set 


### Testing the Model on New Images

#### Five German traffic signs found on the web 

Here are seven German traffic sign images that I found on the web:

: ![alt text][image4]


I cropped them down to only include the signs.  

**Challenges for the model** 
* Image 1,2 and 5 include a lot of background noise.  
* Image 2 has some shadows.  
* Image 4 has a watermark.  
* Image 5 has another sign peeking out in the background 

#### Model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:-------------------------:| 
| Speed Limit 30km/h    | Speed Limit 30km/h 		| 
| Road Work   			| Road Work  				|
| Yield					| Yield						|
| Child Crossing	    | Child Crossing			|
| Turn Right Ahead		| Turn Right Ahead	   		|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

#### Model certainty 

These are the top 5 softmax probabilities for each sign from the model.  

NOTE: Results are rounded to 10 decimal places.  Image 3 shows "100%" for "Yield" and 0 for the rest because the actual amounts were less than 0.0000000001%.

**Image 1: Speed limit (30km/h)**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Speed limit (30km/h)  |  99.6836543083%      |
| Speed limit (20km/h)  |   0.1554675750%      |
| Speed limit (70km/h)  |   0.0655542128%      |
| Speed limit (50km/h)  |   0.0541767222%      |
| Roundabout mandatory  |   0.0112286703%      |


**Image 2: Road work**

| Prediction         			  |     Probability	     | 
|:-------------------------------:|:--------------------:| 
| Road work                       |  99.9952197075%      |
| Dangerous curve to the right    |   0.0029470650%      |
| Road narrows on the right       |   0.0005351747%      |
| End of no passing               |   0.0004788929%      |
| Ahead only                      |   0.0002219612%      |

**Image 3: Yield**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Yield                 | 100.0000000000%      |
| Bumpy road            |   0.0000000000%      |
| Keep left             |   0.0000000000%      |
| Road work             |   0.0000000000%      |
| Ahead only            |   0.0000000000%      |

**Image 4: Children crossing**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Children crossing     |  99.9855160713%      |
| Bicycles crossing     |   0.0081554441%      |
| Bumpy road            |   0.0038773855%      |
| Ahead only            |   0.0009513222%      |
| No entry              |   0.0005606900%      |

**Image 5: Turn right ahead**

| Prediction         	  |     Probability	     | 
|:-----------------------:|:--------------------:|
| Turn right ahead        |  99.9999403954%      |
| Stop                    |   0.0000289722%      |
| Yield                   |   0.0000089241%      |
| Ahead only              |   0.0000061633%      |
| Speed limit (60km/h)    |   0.0000054985%      |


As you can see, the model's certainty for all 5 signs was above 99% for the correct label.  If I had more time, I would have liked to see how it faired with even more challenging images. 
