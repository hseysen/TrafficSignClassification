import os
import numpy as np
import tqdm
import torch
from torchmetrics import F1Score
from torch.utils.data import DataLoader
from datasets.dataset_retrieval import TrafficSignDataset
from models.ResnetModel import Resnet
from models.VGGModel import VGG16
from sklearn.metrics import confusion_matrix


def test(model, data, device):
    f1 = F1Score(num_classes=54, task="multiclass")
    data_iterator = enumerate(data)
    f1_list = []
    f1t_list = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        model.eval()
        tq = tqdm.tqdm(total=len(data))
        tq.set_description("Test:")

        for _, batch in data_iterator:
            # Forward propagation
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            pred = pred.softmax(dim=1)

            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())
            
            y_true.extend(torch.argmax(label, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())

            tq.update(1)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))

    # cm = confusion_matrix(y_true, y_pred)
    # if not os.path.exists("confusion_matrices"):
    #     os.mkdir("confusion_matrices")
    # np.savetxt("confusion_matrices/final.txt", cm, fmt="%d")

    return None

def main():
    device = "cuda:0"

    for data_batch in ["train", "valid", "test"]:
        test_data = TrafficSignDataset(data_batch)
        test_loader = DataLoader(test_data, batch_size=8)

        # VGG Models
        for vggmodel in ["vgg16_sgd", "vgg16_adam"]:
            print(f"Testing {data_batch}, model_{vggmodel}.pth")
            vgg = VGG16(54).to(device)
            checkpoint = torch.load(f"checkpoints/model_{vggmodel}.pth")
            vgg.load_state_dict(checkpoint["state_dict"])
            test(vgg, test_loader, device)
        
        # ResNet Models
        for resnetmodel in ["resnet18_sgd", "resnet18_adam", "final"]:
            print(f"Testing {data_batch}, model_{resnetmodel}.pth")
            resnet = Resnet(54).to(device)
            checkpoint = torch.load(f"checkpoints/model_{resnetmodel}.pth")
            resnet.load_state_dict(checkpoint["state_dict"])
            test(resnet, test_loader, device)


if __name__ == "__main__":
    main()
