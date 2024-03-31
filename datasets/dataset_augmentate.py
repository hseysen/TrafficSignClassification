import os
import random
import numpy as np
import cv2
from PIL import Image, ImageOps


def to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def from_cv2(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


class DataAugmentator:
    def __init__(self):
        self.root = "datasets/TrafficSign"
        self.datadir = os.path.join(self.root, "DATA")

    def rotate_image(self, image, rotation_range=(-15, 15)):
        angle = random.randint(rotation_range[0], rotation_range[1])
        return image.rotate(angle)

    def adjust_perspective(self, image):
        w, h = image.size
        distortion_factor = 0.25
        dx1 = w * random.uniform(-distortion_factor, distortion_factor)
        dx2 = w * random.uniform(-distortion_factor, distortion_factor)
        dy1 = h * random.uniform(-distortion_factor, distortion_factor)
        dy2 = h * random.uniform(-distortion_factor, distortion_factor)

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[dx1, dy1], [w - dx1, dy2], [dx2, h - dy1], [w - dx2, h - dy2]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        orig_image = to_cv2(image)
        transformed_image = cv2.warpPerspective(orig_image, M, (w, h))
        return from_cv2(transformed_image)

    def save_transformed_image(self, image, filename):
        image.save(filename)
    
    def test_one_image(self, test_image_path="datasets/TrafficSign/DATA/0/000_0001.png"):
        # Save the original image for reference
        test_image = Image.open(test_image_path)
        test_image.save("tmp/original_image.png")

        # Test rotation
        result1 = self.rotate_image(test_image)
        result1.save("tmp/rotation_augmentation.png")

        # Test perspective
        result1 = self.adjust_perspective(test_image)
        result1.save("tmp/perspective_augmentation.png")


if __name__ == "__main__":
    DataAugmentator().test_one_image()
    
