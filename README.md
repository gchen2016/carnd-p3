# Carnd-p3: Behavioral Cloning
Training an testing the driving in simulator.

## Overview
 1. Fisrt generate the data using simulator.
 2. Build a CNN to calssify driving.
 3. Test the model in the simulator.
 
## File Structure
 * model.py - The script to create and train the model.
 * drive.py - The script to drive the car.
 * model.json - The model architecture.
 * model.h5 - The model weights.
 * READM.md - This file.
 * The saved file structure from simulator trainning mode is:
```
 driving_log.cvs
 IMG/
   center_yyyy_mm_dd_hh_mm_ss_xxx.jpg
```

 * The driving_log.cvs format is:
```
Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed.
```

## Run Server
 * First run the simulator in the auto mode.
 * python drive.py model.json

## Structure of CNN
The structure is based of the CNN architecture in Nvidia's End to End Learning for Self-Driving Cars parper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


## Envirnment
* Ubuntu 16.04, Kerars, TensorFlow backend
* CPU: Intel i7 6850k
* MEM: 32 G
* GPU: Titian X (Pascal) 12G Mem
* SSD: 0.5 + 1 TB

## Training Approach
### Apporach 1
#### Using the Nvidia's CNN architecture to train the data saved in training mode from the simulator.
Since I am using the GPU, the memory limitation is 12G. The Nvidia's network is too big to fit in. I will have to reduce the size of the network to fit in the GPU memory. I also tried resize the input image size to 84x42, and the origianl network can fit in. But the size is not usable when doing the real testing when connect to the simulator, which has size of 32x160.

### Approach 2
#### Reduce the network architecture by removing 1 FC layer, 2 CNN layer, adjusting size on all remaining layer
##### Network Summary
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 78, 158, 12)   912         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 78, 158, 12)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 77, 24)    7224        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 37, 77, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 35, 75, 36)    7812        activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 35, 75, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 33, 73, 64)    20800       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 33, 73, 64)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 154176)        0           activation_4[0][0]               
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 600)           92506200    flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 600)           0           hidden2[0][0]                    
____________________________________________________________________________________________________
hidden3 (Dense)                  (None, 20)            12020       activation_5[0][0]               
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 20)            0           hidden3[0][0]                    
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             21          activation_6[0][0]               
====================================================================================================
Total params: 92554989
____________________________________________________________________________________________________
```
##### Training Summary
```
Epoch 11/20
5001/5000 [==============================] - 56s - loss: 0.0369 - acc: 0.8494     
Epoch 12/20
5001/5000 [==============================] - 56s - loss: 0.0381 - acc: 0.8486     
Epoch 13/20
5004/5000 [==============================] - 56s - loss: 0.0367 - acc: 0.8507     
Epoch 14/20
5001/5000 [==============================] - 56s - loss: 0.0370 - acc: 0.8468     
Epoch 15/20
5001/5000 [==============================] - 57s - loss: 0.0378 - acc: 0.8502     
Epoch 16/20
5001/5000 [==============================] - 56s - loss: 0.0369 - acc: 0.8482     
Epoch 17/20
5004/5000 [==============================] - 57s - loss: 0.0381 - acc: 0.8465     
Epoch 18/20
5001/5000 [==============================] - 56s - loss: 0.0362 - acc: 0.8496     
Epoch 19/20
5001/5000 [==============================] - 56s - loss: 0.0383 - acc: 0.8492     
Epoch 20/20
5001/5000 [==============================] - 56s - loss: 0.0358 - acc: 0.8506
```
##### Approach Summary
The current input data is a recording of roughly one lap of the trail using keyboard controlling the wheel. After few adjustment on the model and training. The results are similar. The simulator testing shows the car are just moving forward, no turns. It's time to work more on input data.

* load center images into [none, 320, 160, 30] numpy array
* load center angle inot [none] numpy array
* Use keras's ImageDataGenerator to process image like nomalization on the fly. 

* Image isze 320x160 exhausts the GPU memery. Had to resize to 84x32

### Approach 3
#### Record training data
Recoreded serveral data sample, test1, test2, test3, train1, train2, and train3. Sample data is a quick run for code debugging purpose. test1, test2, and test3 are for model testing. Train1, train2, and train3 are for model training and validation in 80/20 splite. Data is recorded using keyboard to steering the wheel. The result is not as smooth as wanted.
#### Training
Using the default Adam optimizer with learning rate 0.02. Data are loaded using Keras ImageDataGenerator. Only use center images and normalized image to RGB ranging from 0 to 1. Epoch was choosed in 5, 20, 50, 100. Training and validation accuracy were around 0.8, and both losese were 0.02. Test was done by actually running the model in the simulator. The model didn't drive the car very far. It could barely made the first left turn.
Next tried incorporrated the left and right images. The steering angles were determined by the ceter image angle toward the center image. This augmentation had great impact on driving the car recovering from off positions. The problem was the data size increased in 3 folds and it cannot be fit into the memory at once. Tried to the flow_from_directoy but no easy way to feed the steering angles. It's time to custom made the gererator.

### Approach N
#### Record more data
Got a gamepad to smooth out the steering angle. Used the old simulator (50Hz) to record images 5 time faster. Recorded 5 data set for 1 loop drive and data set for 4, 5, and 6 loop drive. Recoded sharp turns at high speed. Recorded shparp truns at low speed.
#### Data Augmentation
Only used left and right images. Used to fliped center images left and right, but didn't seem to help.

#### Fine tune the hyper parameters
Since the accuracy and lose were not helping. need to test the model in the simulator a lot. Used plugged in for early stop and check piont. One epoch has all training set of images. Validation set is 5% and turned off later to save compute time. Usually the first epoch check point test better then the follwing epoch ones in the simulator. The better testing didn't show better accuracy or losses, which is weird and remains mystery.

Tried learing rate of 0.001, 0.005, 0.0001, 0.0005, 0.0001, 0.00001, and 0.000001 in different hyper parameters, and data set and size. The 0.001 had the best performance overall. 

#### Other tools to help
Used John Chen' Agile Trainer to spot the trouble location in simulator. The Agile Trainer allows you to override the model using the gamepad and move on the to next cource for the model. I reallied in one model I only interfered the model by slowing down the speed and the model can made the turns needed. The Agile Trainner also sugest using a lower learing rate to train the trouble spots for fine tuning.

I was able to record the immage again with high speed on the sharp turns. The result was the model were able to made those turns at lower speed. Therefor, I changed the drive.py throttle from 0.2 to 0.1 for the model to pass the test. 

### Project Summary
* It's not as strait forward as I were thinking.
* Data is realy realy important
* GPU helps a lot
* Accuracy and loss are not right idication for good train. The actual test in simulation has the final says.
* Shuffling is import to not over train the certain steering.
* Lower learning rate is not better than the right one.
* Driving speed has great impact on the steering.
* It's realy fun project to understain the CNN training for the real world.

## References
### Models
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
* https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
* https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
* https://github.com/jacobgil/keras-steering-angle-visualizations

### Keras
* https://keras.io/
  * https://keras.io/getting-started/faq/
  * https://keras.io/callbacks/

#### Image Process and Augmentation 
* http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

### Simulation Tools
* https://github.com/diyjac/AgileTrainer

## Trouble Shooting
* https://github.com/aymericdamien/TensorFlow-Examples/issues/38
  * InternalError: Dst tensor is not initialized
