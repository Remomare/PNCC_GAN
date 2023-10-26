import torch
import argparse

from models import generator, discriminator, classifier, model_utils

class PNCC_GAN(torch.nn.Module):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        super(PNCC_GAN, self).__init__()
        
        self.z_dim: int = args.z_dim
        self.image_size: int = args.image_size
        self.image_channels: int = args.image_channels
        self.num_class: int = args.num_class
        self.loss_mode: str = args.loss_mode
        
        self.g_learning_rate: float = args.g_learning_rate
        self.d_learning_rate: float = args.d_learning_rate
        
        self.PNCC_Generator = generator.Generator_PNCCGAN(self.z_dim, self.image_size, self.image_channels, self.num_class)
        self.PNCC_Discriminator = discriminator.Discriminator_PNCCGAN(self.image_size, self.image_channels)
        self.PNCC_Classifier = classifier.CNN_Classifier(self.image_channels, self.num_class)
        
        self.d_loss_fu, self.g_loss_fu = model_utils.get_losses_fn(self.loss_mode)
        
        self.g_optimizer = torch.optim.Adam(self.PNCC_Generator.parameters(), lr= self.g_learning_rate)
        self.d_optimizer = torch.optim.Adam(self.PNCC_Discriminator.parameters(), lr= self.d_learning_rate)
        
    def forward():
        pass