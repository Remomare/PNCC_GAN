import torch

class CNN(torch.nn.Module):
    def __init__(self, in_channels: int, num_class: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super(CNN, self).__init__()
        
        self.conv_layer_ls = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, )
        )