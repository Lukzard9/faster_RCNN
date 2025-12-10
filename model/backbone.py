import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, freeze_params=False):
        super(ResNetBackbone, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Stride 2 (640×480) --> (20×15)
        # Stride 1 (640×480) --> (40×30)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        for m in resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.out_channels = 2048
        self.stride = 16

        if freeze_params:
            for param in self.features.parameters():
                param.requires_grad = False
        self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() 
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        
        x = self.features(x)
        return x