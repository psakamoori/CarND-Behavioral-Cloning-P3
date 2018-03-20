Behaviroal Cloing P3: Takes pretrained model file model.h5 and drives the car in simulator in Autonomus mode. Convolutional Neural Network is used to train the model.

### out_video.mp4 shows the network in action.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The Keras implementation of my model can be found in model.py.  

model.h5 is a saved Keras model containing a version of my trained network
that reliably steers the car all the way around the track in my tests.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py:  Keras implementation of model, as well as code to load data and train the model
* drive.py: Connects to Udacity simulator (not provided) to feed image data from the simulator to my model, and angle data from my model back to the simulator
* model.h5:  A saved Keras model, trained using model.py, capable of reliably steering the car all the way around Track 1
* out_video.mp4: Video of the car driving around the track, with steering data supplied by model.h5
README.md

#### 2. Submission includes functional code

How to run : Clone this repository, start the Udacity simulator (not provided), and run below command

python drive.py model.h5

you should see the car drive around the track autonomously without leaving the road.


#### 3. Model.py

Model.py code is reusable. Just change the "mydataIMG" dir path to the path with your images folder.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Model is a Keras implementation of the Nvidia convolutional neural network which trains on Sterring angle and camera sensor image

#### 2. Train, Test Data agumentation and Model parameter tuning

* Used train_test_split to split train and validation data using sklearn python api (80% train and 20% Validation data)
* Flipped image data and sterring measruement for data agumentation
* Shuffled image and sterring data and passed to the model (network)
* One parameter tuned is the Steering angel, read sterring angel provided by simulator and added these values to Left and Right images by adding correction - Tried different correction values (0.5, 0.6, 0.05, 0.06, 0.065). Got lowest loss with 0.065
* Tried different batch_sizes 128 and 32.
* Tried different epochs 3, 5, 10. After epoch 5 validation loss remain almost constant

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

All training was conducted on my laptop with GTX 1060 Nvidia GPU. Struggeld to get tensorflow gpu + keras combination working.
Keras was not picking tensorflow gpu, even though I installed tensorflow-gpu whl file.

Fix is to Install tensorflow-gpu, Install Keras, Uninstall tensorflow-gpu and Install tensorflow-gpu

As explained in the one of the UCity video. Started by training a 1-layer fully connected network, using only data from the center camera, just to get the data pipeline working.  

Next I implemented LeNet in Keras, to see how it would perform. I trained LeNet using only data from the center camera. 
Initially car was moving towards left of the road and finally it went into lake.

Later, I found a bug in my code. I was wrongly storing Sterring values in numpy i.e. I was storing them as continous Center followed by left and right sterring values. Instead I need to place them in interleaved fashion.

Later started a cropping layer as the first layer in my network.  This removed the top 50
and bottom 20 pixels from each input image before passing the image on to the convolution layers.

The top 50 pixels tended to contain sky/trees/horizon, and the bottom 20 pixels contained the car's
hood, all of which are irrelevant to steering and might confuse the model.

Implemented Python generators to serve training and validation data to model.fit_generator(). This helped model.py run faster.
Still Car not following center of the road.

I then implemented the Nvidia neural network architecture found here:
[https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).  

This network is purpose-built for end-to-end training of self-driving car steering based on input from
cameras, so it is ideal for the simulator. 

Finally experimenting correction with different values (0.5, 0.6, 0.05, 0.06, 0.065) and different batch_size, epochs.

#### 2. Final Model Architecture

O/P Size = (Kernel Size - Width/height + 2 * Padding)/(Stride + 1)

Keras automatically computes the output shape of the previous layer/

| Layer                 | Kernel Size |  Stride | Features |  O/P Size  |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                                     
| Cropping              | Crop top 50 pixels and bottom 20 pixels; output shape = 90x320x3 |
| Normalization         | Each new pixel value = old pixel value/255 - 0.5      |

| Convolution 5x5       | 5x5  |  2x2 | 24 | 43x158x24  |
| RELU                  |                               |
| Convolution 5x5       | 5x5  | 2x2  | 36 | 20x77x36   |
| RELU                  |                               |
| Convolution 5x5       | 5x5  | 2x2  | 48 | 8x37x48    |
| RELU                  |                               |
| Convolution 5x5       | 3x3  | 1x1  | 64 | 6x35x64    |
| RELU                  |                               |
| Convolution 5x5       | 3x3  | 1x1  | 64 | 4x33x64    |
| RELU                  |                               |
| Flatten               | Input 4x33x64, output 8448    |
| Fully connected       | Input 8448, output 100        |
| Dropout               | Set units to zero with probability 0.5 |
| Fully connected       | Input 100, output 50          |
| Fully connected       | Input 50, output 10           |
| Fully connected       | Input 10, output 1 (labels)   |

If my layer size math is correct, it does seem like the first fully connected layer has a very large number of parameters
(8448x100) and therefore might overfit.
I added a dropout layer after the first fully connected layer to guard against this possibility.

#### 3. Creation of the Training Set & Training Process

Started UCity windows_sim and collected 18,135 images from all camera sensors.

[Data Collection]: ./images/data_collection.png
[Cetner Sensor Image]: ./images/center_2018_03_18_12_07_38_947.jpg
[Left Sensor Image]: ./images/left_2018_03_18_12_09_30_018.jpg.jpg
[Right Sensor Image]: ./images/right_2018_03_18_12_11_08_656.jpg
