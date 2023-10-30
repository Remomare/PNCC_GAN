import torch
import numpy as np

class Discriminator_PNCCGAN(torch.nn.Module):
    def __init__(self, img_size: int, channels: int, dim: int = 128) -> None:
        super(Discriminator_PNCCGAN, self).__init__()
                
        self.model_ls = torch.nn.Sequential(
            torch.nn.Linear(channels * img_size * img_size, dim * 4),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim * 4, dim * 2),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim * 2, dim),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim, 1),
            torch.nn.Sigmoid()
        )
        
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        logit = self.model_ls(x)
        return logit
    
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


class Discriminator_GAN(torch.nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator_GAN, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
                
        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(self.img_channels * self.img_size * self.img_size), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Discriminator_CGAN(torch.nn.Module):
    def __init__(self, num_classes, img_channels, img_size):
        super(Discriminator_CGAN, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        
        self.label_embedding = torch.nn.Embedding(self.num_classes, self.num_classes)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_classes + int(np.prod(self.img_channels, self.img_size, self.img_size)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Discriminator_WGAN(torch.nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator_WGAN, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(np.prod(self.img_channels, self.img_size, self.img_size)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity