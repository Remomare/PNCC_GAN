import argparse
import torch
import PIL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


 


def mnist_dataset(args:argparse.ArgumentParser) -> DataLoader:
    train_dataset = datasets.MNIST(root= "/MINST", train= True, download= True, transform=transforms.Compose(
                                                            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    test_dataset = datasets.MNIST(root= "/MINST", train= False, transform= transforms.Compose(
                                                            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    train_loader = DataLoader(
        dataset= train_dataset, 
        batch_size= args.batch_size, 
        shuffle=True, 
        num_workers= args.num_workers, 
        pin_memory=args.use_gpu,
        drop_last=True
        )
    return train_loader


                                   