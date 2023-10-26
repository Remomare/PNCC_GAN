from torch.utils.data import DataLoader

from datasets.prepare_data import *

def get_dataset_loader(args, dataset_name:str) -> DataLoader:
    
    if dataset_name == "MNIST":
        return mnist_dataset(args)