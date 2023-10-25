import torch
import PIL
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.Scale(size=(32, 32), interpolation=PIL.Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


def mnist_dataset(args, dataset_path):
    train_dataset = datasets.MNIST(root= dataset_path, train= True, download= True, transform= transform)
    test_dataset = datasets.MNIST(root= dataset_path, train= False, transform= transform)

                                   