The code is organized as following:

1.) challenge.py
Contains SpecPredict class to run as required by problem

2.) classifier.py
The neural network model class

3.) clf_dataset.py
Custom dataset class

4.)train.py
Script to train the model. The script can be run as:
python train.py --train_file 'path_to_train_data_pkl'


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

To evaluate the classification of the solution I used accuracy in the script. However, depending on the distribution of classes, precision or recall may be more appropriate as in real world, there maybe more examples of product and background compared to foreign material.





