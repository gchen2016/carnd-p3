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

## Run Server
 * python drive.py model.json

## Structure of CNN
The structure is based of the CNN architecture in Nvidia's End to End Learning for Self-Driving Cars parper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)



## Training Approach
Using the Nvidia's CNN architecture to train the data saved in training mode from the simulator.

The saved file structure is:
'''
 driving_log.cvs
 IMG/
   center_yyyy_mm_dd_hh_mm_ss_xxx.jpg
'''

The driving_log.cvs format is:
'''
path_to_center.jpg, path_to_left.jpg, path_to_right_img.jpg, center_angle, left_angle, right_angle, a number.
'''

* load center images into [none, 320, 160, 30] numpy array
* load center angle inot [none] numpy array
* Use keras's ImageDataGenerator to process image like nomalization on the fly. 

Then...


## References
### Models
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
* https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

### Keras
* https://keras.io/
  * https://keras.io/getting-started/faq/
  * https://keras.io/callbacks/

#### Image Process and Augmentation 
* http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
