import os
import random
from PIL import Image, ImageOps


class DataAugmentator:
    def __init__(self):
        self.root = "datasets/TrafficSign"
        self.datadir = os.path.join(self.root, "DATA")

    def rotate_image(self, image, rotation_range=(-15, 15)):
        angle = random.randint(rotation_range[0], rotation_range[1])
        return image.rotate(angle)

    def save_transformed_image(self, image, filename):
        image.save(filename)
    
    def test_one_image(self, test_image_path="datasets/TrafficSign/DATA/0/000_1_0001.png"):
        # Save the original image for reference
        test_image = Image.open(test_image_path)
        test_image.save("tmp/original_image.png")

        # Test rotation
        result1 = self.rotate_image(test_image)
        result1.save("tmp/rotation_augmentation.png")


if __name__ == "__main__":
    DataAugmentator().test_one_image()
    
