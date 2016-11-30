# carnd-p3
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
 
## References
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
* https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
