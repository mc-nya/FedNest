# FedNest: Bilevel Federated Learning 

Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The hyper-representation experiments are produced by:
> python [main_hr.py](main_hr.py)

The imbalanced MNIST experiments are produced by:
> python [main_imbalance.py](main_imbalance.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_hr.py --iid --epochs 50 --gpu 0 

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.


