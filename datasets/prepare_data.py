import argparse
import torch
import PIL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose(
    [transforms.Scale(size=(32, 32), interpolation=PIL.Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


def mnist_dataset(args:argparse.ArgumentParser) -> DataLoader:
    train_dataset = datasets.MNIST(root= "/MINST", train= True, download= True, transform= transform)
    test_dataset = datasets.MNIST(root= "/MINST", train= False, transform= transform)
    train_loader = DataLoader(
        dataset= train_dataset, 
        batch_size= args.batch_size, 
        shuffle=True, 
        num_workers= args.num_workers, 
        pin_memory=args.use_gpu,
        drop_last=True
        )
    return train_loader


                                   