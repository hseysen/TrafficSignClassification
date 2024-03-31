import os


class DataSplitter:
    def __init__(self):
        self.root = "datasets/TrafficSign"
        self.datadir = os.path.join(self.root, "DATA")

    def split_dataset(self):
        for cls in os.listdir(self.datadir):
            clsdir = os.path.join(self.datadir, cls)
            print(len(os.listdir(clsdir)), "instances for", cls)


if __name__ == "__main__":
    DataSplitter().view_instances()
    
