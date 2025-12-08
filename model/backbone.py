import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, freeze_params=True):
        super(ResNetBackbone, self).__init__()
        
        # 1. Load the Pretrained ResNet50
        # 'DEFAULT' weights are the modern equivalent of pretrained=True
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        
        
        # --- CHANGE 1: Dilation Strategy ---
        # We replace the stride=2 in layer4 with stride=1 (Dilation)
        # This keeps the feature map larger.
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        # Use dilation=2 for the 3x3 convs in layer4 to maintain receptive field
        for m in resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)

        # 2. Architecture Surgery
        # list(resnet.children())[:-2] grabs everything EXCEPT the last 2 layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 3. Define Output Channels
        # ResNet50's last convolutional block (layer4) has 2048 output channels.
        # We store this variable so the RPN knows how big the input filters are.
        self.out_channels = 2048
        
        # 4. Define Stride
        # ResNet50 reduces dimensions by factor of 32 (Input 640 -> Feat 20)
        self.stride = 16

        # 5. Optional: Freeze Weights
        # Common in exams: "Freeze the backbone so we only train the detector heads"
        if freeze_params:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # 5b. Partial Freezing (Best Practice)
        # Usually we freeze the BatchNorm layers even if we train the rest
        self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        """
        Freezes BatchNorm layers. 
        Batch norms behave weirdly with small batch sizes (common in detection),
        so it's standard practice to freeze them in Faster R-CNN.
        """
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() 
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        Input: Batch of Images (N, 3, 480, 640)
        Output: Feature Maps (N, 2048, 15, 20)
        """
        x = self.features(x)
        return x