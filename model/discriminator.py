import torch

class Discriminator_PNCCGAN(torch.nn.Module):
    def __init__(self, imgae_size: int, channels: int, dim: int = 128) -> None:
        super(Discriminator_PNCCGAN, self).__init__()
                
        self.model_ls = torch.nn.Sequential(
            torch.nn.Linear(channels * imgae_size * imgae_size, dim, dim * 4),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim * 4, dim * 2),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim * 2, dim),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(dim, 1),
            torch.nn.Sigmoid()
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.model_ls(x)
        return logit
    
    
    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                layer.weight.data *= 0.1
                if layer.bias is None:
                    torch.nn.init.constant_(layer.bias, 0)
