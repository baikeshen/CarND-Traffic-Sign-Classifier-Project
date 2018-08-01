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

Here are seven German traffic sign images that I found on the web with minor cropped:

: ![alt text][image4]

Image 1,3, 5 and 6 include a lot of background noise.  It should be relative difficult to classify. Image 6 and Image 7 are cropped from the same image. Image 7 should be easier to identified given less backgroud.

#### Model's predictions

Due to the results from both models are similiar, only the prediction from the first model (Original LeNet model) is shown as below:

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:-------------------------:| 
| Turn Right Ahead		| Turn Right Ahead	   		|
| Speed Limit 60km/h    | Speed Limit 60km/h 		| 
| Child Crossing	    | Child Crossing			|
| Stop Sign	    | Stop Sign			|
| Road Work   			| Road Work  				|
| Speed Limit 20km/h    | Children crossing 		| 
| Speed Limit 20km/h    | Speed Limit 20km/h 		| 



The model was able to correctly guess 6 traffic signs, which gives an accuracy of 85.7% which is lower than the accuracy on the test set of 93.5%

#### Model certainty 

These are the top 5 softmax probabilities for each sign from the model.  

NOTE: Results are rounded to 10 decimal places.  Image 3 shows "100%" for "Yield" and 0 for the rest because the actual amounts were less than 0.0000000001%.

**Image 1: Turn right ahead**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Turn right ahead                        |  99.8363554478% |
| Go straight or left                     |   0.0343522261% |
| No passing for vehicles over 3.5 metric tons  |   0.0254559098% |
| Stop                                    |   0.0245628878% |
| Right-of-way at the next intersection   |   0.0179224138% |


**Image 2: Speed limit (60km/h)**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
|Speed limit (60km/h)                    |  99.9998688698%|
|Turn left ahead                         |   0.0001002719%|
|No entry                                |   0.0000119647%|
|End of all speed and passing limits     |   0.0000105941%|
|Stop                                    |   0.0000080092%|


**Image 3: Children crossing**

| Prediction         			  |     Probability	     | 
|:-------------------------------:|:--------------------:| 
|Children crossing                       |  99.9462306499%|
|End of no passing                       |   0.0346069777%|
|Turn left ahead                         |   0.0102840131%|
|Pedestrians                             |   0.0045954632%|
|Dangerous curve to the right            |   0.0016845350%|

**Image 4: Stop Sign**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
|Stop                                    |  99.9997377396%|
|Speed limit (60km/h)                    |   0.0001563726%|
|Speed limit (30km/h)                    |   0.0000529168%|
|No entry                                |   0.0000396650%|
|Go straight or right                    |   0.0000075150%|


**Image 5: Road work**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
|Road work                               | 100.0000000000%|
|Double curve                            |   0.0000037333%|
|Beware of ice/snow                      |   0.0000012733%|
|Go straight or right                    |   0.0000003162%|
|Yield                                   |   0.0000000627%|

**Image 6: Speed limit (20km/h)**

| Prediction         	  |     Probability	     | 
|:-----------------------:|:--------------------:|
|Children crossing                       |  41.6731387377%|
|Turn left ahead                         |  12.7226799726%|
|Go straight or right                    |   4.4760543853%|
|Speed limit (20km/h)                    |   3.7788741291%|
|End of no passing                       |   3.2620340586%|

**Image 7: Speed limit (20km/h)**

| Prediction         	  |     Probability	     | 
|:-----------------------:|:--------------------:|
|Speed limit (20km/h)                    |  76.8208980560%|
|Speed limit (30km/h)                    |   8.7321907282%|
|Roundabout mandatory                    |   7.4885345995%|
|End of all speed and passing limits     |   2.7689540759%|
|Go straight or left                     |   1.6382671893%|

As you can see, the model's certainty for the first 5 signs was above 99% for the correct label.  However,, it is quite challenge to classify the sign of speed limit （20km/h). One possible reason is lack of training samples (less than 250). If applying more additional sample for speed limit (20km/h), the accuracy could be improved.
