import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights
from models.bayesian_layers import BayesianLinear, BayesianConv2d


class BayesianEfficientNetFC(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(BayesianEfficientNetFC, self).__init__()
        # Load EfficientNet without the classifier

        # Use pre-trained weights or start from scratch
        if pretrained:
            self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load pre-trained weights
        else:
            self.efficientnet = efficientnet_b0(weights=None)  # Start from scratch
        
        self.efficientnet.classifier = nn.Sequential()  # Remove the classifier
        self.bayesian_fc = BayesianLinear(1280, num_classes)  # Bayesian linear layer

    def forward(self, x):
        x = self.efficientnet(x)  # EfficientNet forward pass
        x = self.bayesian_fc(x)  # Bayesian classifier
        
        return x
    
    def kl_divergence(self):
        return self.bayesian_fc.kl_divergence()
    

class BayesianEfficientNetConv(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(BayesianEfficientNetConv, self).__init__()
        
        # Use pre-trained weights or start from scratch
        if pretrained:
            self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load pre-trained weights
        else:
            self.efficientnet = efficientnet_b0(weights=None)  # Start from scratch

        self.efficientnet.features[0][0] = BayesianConv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Replace first Conv2D layer with Bayesian
        
        # Additional Bayesian layers could be inserted, e.g., later in the features block
        # self.efficientnet.features[2][0] = BayesianConv2d(24, 40, kernel_size=3, stride=2, padding=1)
        
        self.efficientnet.classifier = BayesianLinear(1280, num_classes)  # Replace the classifier with Bayesian layer

    def forward(self, x):
        x = self.efficientnet(x)  # EfficientNet forward pass
        return x
    
    def kl_divergence(self):
        kl = self.bayesian_fc.kl_divergence()  # KL divergence for the Bayesian FC layer
        
        # Add KL divergence from Bayesian convolutional layers
        for module in self.efficientnet.features:
            for submodule in module:
                if isinstance(submodule, BayesianConv2d):
                    kl += submodule.kl_divergence()
        return kl
    

# Define a ResNet with a Bayesian layer
class BayesianResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(BayesianResNet, self).__init__()
        
        # Use pre-trained weights or start from scratch
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)  # Start from scratch
        
        self.resnet.fc = BayesianLinear(512, num_classes)  # Replace final FC layer with Bayesian layer

    def forward(self, x):
        x = self.resnet(x)  # ResNet forward pass
        return x
    
    def kl_divergence(self):
        # KL divergence from the Bayesian layer
        return self.resnet.fc.kl_divergence()