# carnd-p3 (In Progress ...)
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



* load center images into [none, 320, 160, 30] numpy array
* load center angle inot [none] numpy array
* Use keras's ImageDataGenerator to process image like nomalization on the fly. 

* Image isze 320x160 exhausts the GPU memery. Had to resize to 84x32


## References
### Models
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
* https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
* https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

### Keras
* https://keras.io/
  * https://keras.io/getting-started/faq/
  * https://keras.io/callbacks/

#### Image Process and Augmentation 
* http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

## Trouble Shooting
* https://github.com/aymericdamien/TensorFlow-Examples/issues/38
  * InternalError: Dst tensor is not initialized
