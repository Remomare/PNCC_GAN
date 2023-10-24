import torch

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
        
        self.init_weights()
    
                        
    def forward(self, z: torch.Tensor, prev_gen_class: list) -> torch.Tensor:
        input = torch.cat([z, self.prev_gen_class_embedding(prev_gen_class)], dim=1)
        x = self.model_ls(input.view(input.size(0), input.size(1), 1, 1))
        return x
        
            
        
    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                layer.weight.data *= 0.1
                if layer.bias is None:
                    torch.nn.init.constant_(layer.bias, 0)
