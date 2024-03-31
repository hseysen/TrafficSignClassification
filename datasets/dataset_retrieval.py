import os
import numpy as np
from PIL import Image
import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficSignDataset(Dataset):
    def __init__(self, mode="train", root="datasets/TrafficSign", transforms=None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms

        # Select split
        self.folder = os.path.join(self.root, self.mode.upper())

        # Initialize lists
        self.image_list = []
        self.label_list = []

        # Save class lists
        self.class_list = os.listdir(self.folder)
        self.class_list.sort()

        # Loop over every image and add the corresponding entry to image and label lists
        for class_id in range(len(self.class_list)):
            for image in os.listdir(os.path.join(self.folder, self.class_list[class_id])):
                self.image_list.append(os.path.join(self.folder, self.class_list[class_id], image))
                label = np.zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)

    def __getitem__(self, index):
        # Retrieve the entry at requested index
        image_name = self.image_list[index]
        label = self.label_list[index]

        # Convert label to a tensor, and apply transforms to the image
        label = torch.tensor(label)
        image = Image.open(image_name)
        if(self.transforms):
            image = self.transforms(image)
        else:
            tr = transforms.Compose([transforms.ToTensor()])
            image = tr(image)

        return image, label

    def __len__(self):
        return len(self.image_list)
