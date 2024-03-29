import os
import random
from PIL import Image, ImageOps


class DataAugmentator:
    def __init__(self):
        self.root = "datasets/TrafficSign"
        self.datadir = os.path.join(self.root, "DATA")

    def rotate_image(self, image, rotation_range=(-55, 55)):
        angle = random.randint(rotation_range[0], rotation_range[1])
        return image.rotate(angle)

    def scale_image(self, image, scale_range=(0.75, 1.25)):
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return image.resize((new_width, new_height))

    def save_transformed_image(self, image, filename):
        image.save(filename)


if __name__ == "__main__":
    print("unimplemented")
    
