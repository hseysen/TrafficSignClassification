from operator import add
from PIL import ImageStat
import torch
from torchvision.transforms import functional
from datasets.dataset_retrieval import TrafficSignDataset


NORMS_FILE = "norms.txt"


# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/13
class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))


def calculate_norms(dataset):
    dl = torch.utils.data.DataLoader(dataset, batch_size=30, num_workers=6)
    
    stats = None
    for image, _ in dl:
        for i in range(image.shape[0]):
            s = Stats(functional.to_pil_image(image[i]))
            if stats is None:
                stats = s
            else:
                stats += s

    mean = stats.mean
    stdv = stats.stddev
    with open(NORMS_FILE, "w") as wf:
        for x1, x2 in zip(mean, stdv):
            print(f"{x1}\t{x2}", file=wf)


def retrieve_norms():
    mean = []
    stdv = []
    with open(NORMS_FILE, "r") as rf:
        for line in rf.readlines():
            x1, x2 = map(float, line.split())
            mean.append(x1)
            stdv.append(x2)
    return mean, stdv


if __name__ == "__main__":
    calculate_norms(TrafficSignDataset())
    retrieved_mean, retrieved_stdv = retrieve_norms()
    print(retrieved_mean, retrieved_stdv, sep="\n")
