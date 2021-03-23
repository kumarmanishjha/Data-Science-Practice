
CODE ORGANIZATION:
-------------------
The code is organized as following:

1.) challenge.py
Contains SpecPredict class to run as required by problem

2.) classifier.py
The neural network model class

3.) clf_dataset.py
Custom dataset class

4.)train.py
Script to train the model.

challenge.ipynb is the code where I first wrote the program and modularized it into separate .py files. 

HOW TO RUN?
------------
1.) To train the model
$python train.py --train_file 'path_to_train_data_pkl'

2.) To test a sample run
$python run_test.py


MODEL ARCHITECTURE:
--------------------
To train the model I used a simple MLP whose architecture is given below:
Net(
  (layer_1): Linear(in_features=502, out_features=512, bias=True)
  (layer_2): Linear(in_features=512, out_features=128, bias=True)
  (layer_3): Linear(in_features=128, out_features=64, bias=True)
  (layer_out): Linear(in_features=64, out_features=3, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.4, inplace=False)
  (batchnorm1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (batchnorm2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (batchnorm3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
The model consists of 3 layer MLP with batchnormalization and RELU activation layers along with 40% dropout rate.

EVALUATION:
------------
To evaluate the classification of the solution I used accuracy in the script. However, depending on the distribution of classes, precision or recall may be more appropriate as in real world, there maybe higher number of samples of data from the product(label 0) and background(label 1) compared to foreign material.

SPEED OPTIMIZATION:
-------------------
The current model has inference time for 10000 samples as:
On GPU: 0.26 seconds
On CPU: 0.27 seconds

We can observe that model is fast on both CPU and GPU and the reason is that it's a relatively small model. Smaller models are good for IOT devices where inference is needed in realtime with high accuracy. Quantization is available in pytorch to run the inference on 8-bit integer weights which reduce the model 4 times and make the inference faster. However, this comes at a cost of accuracy. We can choose to experiment with 16-bit or 8-bit and choose the model with better accuracy and acceptable size.







