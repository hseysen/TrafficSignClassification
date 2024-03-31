import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, class_count)

    def forward(self, image):
        return self.vgg16(image)
