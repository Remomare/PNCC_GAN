import torch
import numpy as np

class Generator_PNCCGAN(torch.nn.Module):
    
    def __init__(self, z_dim: int, image_size: int , img_channels: int, num_classes: int, dim : int = 128) -> None:
        super(Generator_PNCCGAN, self).__init__()
        self.z_dim = z_dim
        self.imgae_size = image_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        
        self.prev_gen_class_embedding = torch.nn.Embedding(num_classes, num_classes)
        
        self.model_ls = torch.nn.Sequential(
            torch.nn.Linear(z_dim + num_classes, dim),
            torch.nn.ReLU(),
            
            torch.nn.Linear(dim, dim * 2),
            torch.nn.BatchNorm1d(dim * 2),
            torch.nn.ReLU(),
            
            torch.nn.Linear(dim * 2, dim * 4),
            torch.nn.BatchNorm1d(dim * 4),
            torch.nn.ReLU(),
            
            torch.nn.Linear(dim * 4, dim * 8),
            torch.nn.BatchNorm1d(dim * 8),
            torch.nn.ReLU(),
            
            torch.nn.Linear(dim * 8, img_channels * image_size * image_size),
            torch.nn.Tanh()
        )
        
        self._init_weights()
    
                        
    def forward(self, z: torch.Tensor, prev_gen_class: list) -> torch.Tensor:
        input = torch.cat([z, self.prev_gen_class_embedding(prev_gen_class)], dim=1)
        x = self.model_ls(input.view(input.size(0), input.size(1), 1, 1))
        return x
        
            
        
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

class Generator_GAN(torch.nn.Module):
    def __init__(self, z_dim, img_channels, img_size):
        super(Generator_GAN, self).__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(self.z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(np.prod(self.img_channels, self.img_size, self.img_size))),
            torch.nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img


class Generator_CGAN(torch.nn.Module):
    def __init__(self, z_dim, img_channels, img_size, num_classes):
        super(Generator_CGAN, self).__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.label_emb = torch.nn.Embedding(self.num_classes, self.num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(self.z_dim + self.num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(np.prod(self.img_channels, self.img_size, self.img_size))),
            torch.n.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img


class Generator_WGAN(torch.nn.Module):
    def __init__(self, z_dim, img_channels, img_size):
        super(Generator_WGAN, self).__init__()
        self. z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(self.z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(np.prod(self.img_channels, self.img_size, self.img_size))),
            torch.nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], self.img_channels, self.img_size, self.img_size)
        return img
