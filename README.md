# Carnd-P3: Behavioral Cloning
Training an testing the driving in simulator.

## Overview
 1. First generate the data using simulator.
 2. Build a CNN to classify driving.
 3. Test the model in the simulator.
 
## File Structure
 * model.py - The script to create and train the model.
 * drive.py - The script to drive the car.
 * model.json - The model architecture.
 * model.h5 - The model weights.
 * gen_dir_data.py - The custom image generator
 * READM.md - This file.
 * The saved file structure from simulator training mode is:
```
 driving_log.csv
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


## Environment
* Ubuntu 16.04, Keras, TensorFlow backend
* CPU: Intel i7 6850k
* MEM: 32 G
* GPU: Titian X (Pascal) 12G Mem
* SSD: 0.5 + 1 TB

## Training Approach
### Approach 1
#### Using the Nvidia's CNN architecture to train the data saved in training mode from the simulator.
Since I am using the GPU, the memory limitation is 12G. The Nvidia's network is too big to fit in. I will have to reduce the size of the network to fit in the GPU memory. I also tried resize the input image size to 84x42, and the origianl network can fit in. But the size is not usable when doing the real testing when connect to the simulator, which has size of 32x160.

### Approach 2
#### Architect
Reduce the network architecture by removing 1 FC layer, 2 CNN layer, adjusting size on all remaining layer
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
* load center angle into [none] numpy array
* Use keras's ImageDataGenerator to process image like normalization on the fly. 

* Image isze 320x160 exhausts the GPU memery. Had to resize to 84x32

### Approach 3
#### Record training data
Recorded several data sample, test1, test2, test3, train1, train2, and train3. Sample data is a quick run for code debugging purpose. test1, test2, and test3 are for model testing. Train1, train2, and train3 are for model training and validation in 80/20 splite. Data is recorded using keyboard to steering the wheel. The result is not as smooth as wanted.
#### Training
Using the default Adam optimizer with learning rate 0.02. Data are loaded using Keras ImageDataGenerator. Only use center images and normalized image to RGB ranging from 0 to 1. Epoch was choosed in 5, 20, 50, 100. Training and validation accuracy were around 0.8, and both losese were 0.02. Test was done by actually running the model in the simulator. The model didn't drive the car very far. It could barely made the first left turn.
Next tried incorporated the left and right images. The steering angles were determined by the center image angle toward the center image. This augmentation had great impact on driving the car recovering from off positions. The problem was the data size increased in 3 folds and it cannot be fit into the memory at once. Tried to the flow_from_directory but no easy way to feed the steering angles. It's time to custom made the generator.

### Approach N

#### Architect
Reduced Nvidia's End-to-end model:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 78, 158, 12)   912         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 78, 158, 12)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 77, 24)    7224        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 37, 77, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 12, 25, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 23, 36)    7812        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10, 23, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 3, 7, 36)      0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 5, 64)      20800       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 1, 5, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 320)           0           activation_4[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 320)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 600)           192600      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 600)           0           hidden2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 600)           0           activation_5[0][0]               
____________________________________________________________________________________________________
hidden3 (Dense)                  (None, 20)            12020       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 20)            0           hidden3[0][0]                    
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             21          activation_6[0][0]               
====================================================================================================
Total params: 241389
____________________________________________________________________________________________________
```
Since the simulator is a simplified real world, reduced the model parameters can save space and time for computation.
Adds a lambda input layer for normalizing RGB data to range (-1, 1). Used image input size as is for 320x160. Had 2x2 stride in the first two CNN layer to reduce spacial size faster, instead of resize the images, to save footprint.
Maxim pooling added to the CNN 2 layer for the same purpose. Used stride (1, 1) for CNN 3, 4 layers for keeping small special size but increased depth to 36, and 64 for more features.
There are 3 fully connected layers with 2 50% dropout layer to prevent overfitting. The output layer was one neuron with tanh activation for regression prediction in range (-1, 1) for the steering angle.    

#### Record more data
Got a game pad to smooth out the steering angle. Used the old simulator (50Hz) to record images 5 time faster. Recorded 5 data set for 1 loop drive and data set for 4, 5, and 6 loop drive. Recoded sharp turns at high speed. Recorded shparp truns at low speed.

#### Data Augmentation
Only used left and right images. Used to flipped center images left and right, but didn't seem to help.

#### Fine tune the hyper parameters
Since the accuracy and lose were not helping. need to test the model in the simulator a lot. Used plugged in for early stop and check piont. One epoch has all training set of images. Validation set is 5% and turned off later to save compute time. Usually the first epoch check point test better then the follwing epoch ones in the simulator. The better testing didn't show better accuracy or losses, which is weird and remains mystery.

Tried learning rate of 0.001, 0.005, 0.0001, 0.0005, 0.0001, 0.00001, and 0.000001 in different hyper parameters, and data set and size. The 0.001 had the best performance overall. 

#### Other tools to help
Used John Chen' Agile Trainer to spot the trouble location in simulator. The Agile Trainer allows you to override the model using the gamepad and move on the to next cource for the model. I reallied in one model I only interfered the model by slowing down the speed and the model can made the turns needed. The Agile Trainner also sugest using a lower learing rate to train the trouble spots for fine tuning.

I was able to record the image again with high speed on the sharp turns. The result was the model were able to made those turns at lower speed. Therefor, I changed the drive.py throttle from 0.2 to 0.1 for the model to pass the test.

##### Testing in the simlulator from Youtube
[Track 1 learning rate 0.001 epcho 1] (https://www.youtube.com/watch?v=2ORa0psALqc)

### Lesson Learned
#### How to make sharp turns
When we make the sharp turns during the recording, we tend to slow down the speed and make the smooth turns with SMALLER angles. Since there is no speed information in the model, the smaller angles learned in the lower speed won't be able to make turns in the testing, which is faster, speed. I was able to record the sharp turns in the same or higher speed, related to the more straight line. The data fed into the model to make those sharp turns. Testing in lower speed, like throttle=0.1, instead of 0.2, in drive.py helps those turns. Then We can fine turn and increase the testing speed if needed.
#### Train each image only once
I have my image generator to produce full length of  the number of images per epoch. I have trained the model with many epochs with check point enabled for each one. The test results in running the simulator tell me that the first epoch mostly yellsthe best result for the training from the scratch. The conclusion are observed over combinations of different set of data, learning rates, augmentations, and hyper parameters. Despite the output of the training accuracy and loss, the first epoch yells the best result over the following ones. 

This leads the mystery for the accuracy and loss, which are not related to the performance of the model at all.

#### Make sure data augmentation is sound
Whe appied left and right images, first I did was add and subtract a contant number from center image angle. After tried serverl nubmers, the perfomance in the simulator did not improved, or getting worse. Then I switched to the current factory formula toward the center, and settle on the fator of 0.75. The perfomance is consitantly good during differnt test.

#### Change only one thing at time
Left and  right image gave me a lesson for data augmentation: make sure data is "good enough" before and after augmentation. Otherwise it will just give you more headache. :

####The Art of Empirical
All the great theories can give you a jump start to implementation. When we get stuck, we will need to think outside of the box and throw alway those theories one by one with testing. If we are insist on the accuracy or loss without testing the model in the real environments, we would have lost the correct experiments and gone through endless wrong paths.

##### Training and validation losses vs performance in simulator
I did train a model with learning rate chaged to 0.0001 for 20 epochs. Validation set at 3000 images for about 5% of the all data. The lowest traing and validation losses were epoch 19, and 20. But the best performance during the testing in the simulator was epoch 9, which had higher training and validation losses!
Epoch 20 ran into the right side of the lake in the first left turn. Epoch 1 ran into the left rail of the bridge at the first laps. On the other hand, epoch 9 stayed on the tracks for more than 10 laps and over 15 minutes.

Here is the output of the training:

```
$ time python3 model_1.py
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
Train3: None lr= 0.0001 epochs= 20
Train3: data_dirs=['data/record/50hz/gfull1/']
init_model1
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 78, 158, 12)   912         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 78, 158, 12)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 77, 24)    7224        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 37, 77, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 12, 25, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 23, 36)    7812        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10, 23, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 3, 7, 36)      0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 5, 64)      20800       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 1, 5, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 320)           0           activation_4[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 320)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 600)           192600      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 600)           0           hidden2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 600)           0           activation_5[0][0]               
____________________________________________________________________________________________________
hidden3 (Dense)                  (None, 20)            12020       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 20)            0           hidden3[0][0]                    
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             21          activation_6[0][0]               
====================================================================================================
Total params: 241389
____________________________________________________________________________________________________

Shuffling index ...
DataGen train size=66325 valid size=3491
Epoch 1/20
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:02:00.0
Total memory: 11.90GiB
Free memory: 11.26GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:02:00.0)
66000/66325 [============================>.] - ETA: 0s - loss: 0.0715 - acc: 0.8619  
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0714 - acc: 0.8620/usr/local/lib/python3.5/dist-packages/keras/engine/training.py:1470: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
Epoch 00000: saving model to ckpt/50hz-0.0001-00.h5
66400/66325 [==============================] - 72s - loss: 0.0714 - acc: 0.8620 - val_loss: 0.0737 - val_acc: 0.8585
Epoch 2/20
65800/66325 [============================>.] - ETA: 0s - loss: 0.0633 - acc: 0.8611 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0633 - acc: 0.8612Epoch 00001: saving model to ckpt/50hz-0.0001-01.h5
66400/66325 [==============================] - 69s - loss: 0.0633 - acc: 0.8610 - val_loss: 0.0690 - val_acc: 0.8556
Epoch 3/20
65600/66325 [============================>.] - ETA: 0s - loss: 0.0602 - acc: 0.8584 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0602 - acc: 0.8584Epoch 00002: saving model to ckpt/50hz-0.0001-02.h5
66400/66325 [==============================] - 65s - loss: 0.0602 - acc: 0.8583 - val_loss: 0.0665 - val_acc: 0.8502
Epoch 4/20
65400/66325 [============================>.] - ETA: 0s - loss: 0.0587 - acc: 0.8565 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0587 - acc: 0.8564Epoch 00003: saving model to ckpt/50hz-0.0001-03.h5
66400/66325 [==============================] - 68s - loss: 0.0588 - acc: 0.8565 - val_loss: 0.0652 - val_acc: 0.8499
Epoch 5/20
65200/66325 [============================>.] - ETA: 1s - loss: 0.0572 - acc: 0.8559 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0572 - acc: 0.8558Epoch 00004: saving model to ckpt/50hz-0.0001-04.h5
66400/66325 [==============================] - 68s - loss: 0.0573 - acc: 0.8558 - val_loss: 0.0641 - val_acc: 0.8496
Epoch 6/20
65000/66325 [============================>.] - ETA: 1s - loss: 0.0559 - acc: 0.8553 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0560 - acc: 0.8552Epoch 00005: saving model to ckpt/50hz-0.0001-05.h5
66400/66325 [==============================] - 71s - loss: 0.0560 - acc: 0.8552 - val_loss: 0.0632 - val_acc: 0.8499
Epoch 7/20
64800/66325 [============================>.] - ETA: 1s - loss: 0.0546 - acc: 0.8554 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0548 - acc: 0.8552Epoch 00006: saving model to ckpt/50hz-0.0001-06.h5
66400/66325 [==============================] - 71s - loss: 0.0547 - acc: 0.8552 - val_loss: 0.0621 - val_acc: 0.8490
Epoch 8/20
64600/66325 [============================>.] - ETA: 1s - loss: 0.0536 - acc: 0.8547 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0537 - acc: 0.8547Epoch 00007: saving model to ckpt/50hz-0.0001-07.h5
66400/66325 [==============================] - 71s - loss: 0.0537 - acc: 0.8546 - val_loss: 0.0611 - val_acc: 0.8505
Epoch 9/20
64400/66325 [============================>.] - ETA: 1s - loss: 0.0522 - acc: 0.8553 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0522 - acc: 0.8553Epoch 00008: saving model to ckpt/50hz-0.0001-08.h5
66400/66325 [==============================] - 70s - loss: 0.0522 - acc: 0.8551 - val_loss: 0.0601 - val_acc: 0.8505
Epoch 10/20
64200/66325 [============================>.] - ETA: 2s - loss: 0.0507 - acc: 0.8553 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0507 - acc: 0.8552Epoch 00009: saving model to ckpt/50hz-0.0001-09.h5
66400/66325 [==============================] - 70s - loss: 0.0508 - acc: 0.8551 - val_loss: 0.0589 - val_acc: 0.8516
Epoch 11/20
64000/66325 [===========================>..] - ETA: 2s - loss: 0.0495 - acc: 0.8553 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0495 - acc: 0.8551Epoch 00010: saving model to ckpt/50hz-0.0001-10.h5
66400/66325 [==============================] - 69s - loss: 0.0495 - acc: 0.8552 - val_loss: 0.0580 - val_acc: 0.8516
Epoch 12/20
63800/66325 [===========================>..] - ETA: 2s - loss: 0.0483 - acc: 0.8561 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0484 - acc: 0.8559Epoch 00011: saving model to ckpt/50hz-0.0001-11.h5
66400/66325 [==============================] - 68s - loss: 0.0484 - acc: 0.8561 - val_loss: 0.0574 - val_acc: 0.8533
Epoch 13/20
63600/66325 [===========================>..] - ETA: 2s - loss: 0.0468 - acc: 0.8564 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0468 - acc: 0.8563Epoch 00012: saving model to ckpt/50hz-0.0001-12.h5
66400/66325 [==============================] - 71s - loss: 0.0468 - acc: 0.8563 - val_loss: 0.0562 - val_acc: 0.8548
Epoch 14/20
63400/66325 [===========================>..] - ETA: 3s - loss: 0.0455 - acc: 0.8568 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0455 - acc: 0.8568Epoch 00013: saving model to ckpt/50hz-0.0001-13.h5
66400/66325 [==============================] - 70s - loss: 0.0455 - acc: 0.8567 - val_loss: 0.0549 - val_acc: 0.8545
Epoch 15/20
63200/66325 [===========================>..] - ETA: 3s - loss: 0.0444 - acc: 0.8570 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0444 - acc: 0.8570Epoch 00014: saving model to ckpt/50hz-0.0001-14.h5
66400/66325 [==============================] - 70s - loss: 0.0444 - acc: 0.8568 - val_loss: 0.0533 - val_acc: 0.8551
Epoch 16/20
63000/66325 [===========================>..] - ETA: 3s - loss: 0.0430 - acc: 0.8580 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0431 - acc: 0.8577Epoch 00015: saving model to ckpt/50hz-0.0001-15.h5
66400/66325 [==============================] - 64s - loss: 0.0430 - acc: 0.8578 - val_loss: 0.0524 - val_acc: 0.8551
Epoch 17/20
62800/66325 [===========================>..] - ETA: 3s - loss: 0.0416 - acc: 0.8596 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0416 - acc: 0.8593Epoch 00016: saving model to ckpt/50hz-0.0001-16.h5
66400/66325 [==============================] - 68s - loss: 0.0416 - acc: 0.8594 - val_loss: 0.0519 - val_acc: 0.8531
Epoch 18/20
62600/66325 [===========================>..] - ETA: 3s - loss: 0.0408 - acc: 0.8592 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0407 - acc: 0.8592Epoch 00017: saving model to ckpt/50hz-0.0001-17.h5
66400/66325 [==============================] - 72s - loss: 0.0406 - acc: 0.8592 - val_loss: 0.0508 - val_acc: 0.8505
Epoch 19/20
62400/66325 [===========================>..] - ETA: 4s - loss: 0.0398 - acc: 0.8602 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0398 - acc: 0.8600Epoch 00018: saving model to ckpt/50hz-0.0001-18.h5
66400/66325 [==============================] - 71s - loss: 0.0397 - acc: 0.8600 - val_loss: 0.0497 - val_acc: 0.8533
Epoch 20/20
62200/66325 [===========================>..] - ETA: 4s - loss: 0.0388 - acc: 0.8606 
Shuffling index ...
66200/66325 [============================>.] - ETA: 0s - loss: 0.0387 - acc: 0.8606Epoch 00019: saving model to ckpt/50hz-0.0001-19.h5
66400/66325 [==============================] - 72s - loss: 0.0387 - acc: 0.8606 - val_loss: 0.0499 - val_acc: 0.8545

Model saved to model.json and model.h5

real	23m27.874s
user	28m14.752s
sys	2m35.624s

```

### Project Summary
* It's not as strait forward as I were thinking.
* Data is really really important
* GPU helps a lot
* Accuracy and loss are not right indication for good train. The actual test in simulation has the final says.
* Shuffling is import to not over train the certain steering.
* Lower learning rate is not better than the right one.
* Driving speed has great impact on the steering.
* It's a fun project to understand the CNN training for the approximated real world.

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


