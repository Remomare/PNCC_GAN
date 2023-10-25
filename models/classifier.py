import torch

class CNN_Classifier(torch.nn.Module):
    def __init__(self, in_channels: int, num_class: int) -> None:
        super(CNN_Classifier, self).__init__()
        
        self.keep_prob: float = 0.5
        
        self.conv_layer_ls = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size= 3, stride= 1, padding= 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride= 2),
            
            torch.nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride= 2),
                        
            torch.nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding= 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride= 2, padding=1)
        )
        self.fc_layer_ls = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 1 - self.keep_prob),
            torch.nn.Linear(625, num_class, bias=True)
        )
        
        self._init_weights()
    
    def forward(self, x):
        y = self.conv_layer_ls(x)
        y = y.view(y.size(0), -1)
        y = self.fc_layer_ls(y)
        return y
    
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
