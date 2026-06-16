import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(ResNet, self).__init__()
        
        if depth == 50:
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
        elif depth == 101:
            if pretrained:
                self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            else:
                self.backbone = models.resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Remove the final classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classification layer
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        
        if return_features:
            return features, self.fc(features)
        else:
            return self.fc(features)
    
    def get_features(self, x):
        """Extract features without classification"""
        return self.backbone(x)


class Classifier(nn.Module):
    """Classifier for loading pre-trained source models"""
    def __init__(self, args, checkpoint_path=None):
        super(Classifier, self).__init__()
        
        if args.domain == 'visda17':
            self.backbone = ResNet(101, args.num_classes, pretrained=False)
        else:
            self.backbone = ResNet(50, args.num_classes, pretrained=False)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load pre-trained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=True)
        self.backbone.load_state_dict(checkpoint, strict=False)
    
    def forward(self, x, return_features=False):
        return self.backbone(x, return_features)
    
    def eval(self):
        self.backbone.eval()
        return self
    
    def cuda(self):
        self.backbone.cuda()
        return self