import torch
import generator
import discriminator
import classifier

class PNCC_GAN(torch.nn.Module):
    def __init__(self) -> None:
        super(PNCC_GAN, self).__init__()
        
        self.PNCC_Generator = generator.Generator_PNCCGAN()
        self.PNCC_Discriminator = discriminator.Discriminator_PNCCGAN()
        self.PNCC_Classifier = classifier.CNN_Classifier()
        
    def forward():
        pass