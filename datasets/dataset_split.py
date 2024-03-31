import os
import random
from PIL import Image


class DataSplitter:
    def __init__(self):
        self.root = "datasets/TrafficSign"
        self.datadir = os.path.join(self.root, "DATA")
        self.traindir = os.path.join(self.root, "TRAIN")
        self.validdir = os.path.join(self.root, "VALID")

    def split_dataset(self):
        for cls in os.listdir(self.datadir):
            class_traindir = os.path.join(self.traindir, str(cls))
            class_validdir = os.path.join(self.validdir, str(cls))
            if not os.path.exists(class_traindir):
                os.mkdir(class_traindir)
            if not os.path.exists(class_validdir):
                os.mkdir(class_validdir)

            clsdir = os.path.join(self.datadir, cls)
            instance_count = len(os.listdir(clsdir))
            train_split = int(instance_count * 0.8)

            random.seed(42)
            images = os.listdir(clsdir)
            random.shuffle(images)

            for image in images[:train_split]:
                imagedir = os.path.join(clsdir, image)
                Image.open(imagedir).save(os.path.join(class_traindir, image))

            for image in images[train_split:]:
                imagedir = os.path.join(clsdir, image)
                Image.open(imagedir).save(os.path.join(class_validdir, image))
                

if __name__ == "__main__":
    DataSplitter().split_dataset()
    
