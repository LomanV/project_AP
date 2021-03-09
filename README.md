# project_AP
Deep learning project on mixup training

The primary (and the only source of the code) is mixup-2.ipynb jupyter notebook in the repo.
There is Net class in the notebook which implements mixup training on a dataset of your choice (suitable for at least CIFAR-10 and MNIST), using resnet18 or wide_resnet50_2 architectures from pytorch.
Methods of the class include 'mixup' and 'mixup_criterion' (implementation of mixup itself and evaluation of model's prediction respectively), 'train' and 'test' (which are just train and test loops) and 'predict' for single sample prediction.
There are also separate functions that are used to actually evaluate model's calibration and plot the results. These include 'compute_ece_loss', 'plot_con_acc' (confidence/accuracy plot) and 'evaluate_calibration', which is the primary calibration evaluation function (previous ones are being called inside of it). 
