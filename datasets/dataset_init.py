from torch.utils.data import DataLoader

from datasets.prepare_data import *

def dataset_init(args, dataset_name:str) -> DataLoader:
    
    if dataset_name == "MNIST":
        return mnist_dataset(args)