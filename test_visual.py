import numpy as np
import pandas as pd
import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
from torchmetrics import F1Score
from torch.utils.data import DataLoader
from datasets.dataset_retrieval import TrafficSignDataset
from torchvision.transforms import functional as tf
from models.ResnetModel import Resnet


label_dict = pd.read_csv("datasets/TrafficSign/labels.csv")
stats = dict()


def add_margin_and_text(image, text):
    w, h = image.size
    new_h = h + h // 2
    new_image = Image.new(image.mode, (w, new_h), color="white")
    new_image.paste(image, (0, h // 2))
    
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()
    text_position = (15, 15)
    draw.text(text_position, text, fill="black", font=font)
    
    return new_image


def test(model, data, device):
    f1 = F1Score(num_classes=54, task="multiclass")
    data_iterator = enumerate(data)
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()
        tq = tqdm.tqdm(total=len(data))
        tq.set_description("Test:")

        for _, batch in data_iterator:
            # Forward propagation
            image, label = batch
            i_cpy = tf.to_pil_image(image[0])
            l_cpy = np.argmax(label[0])
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            pred = pred.softmax(dim=1)
            p_cpy = torch.argmax(pred)
            
            act_cls = int(classes[l_cpy])
            prd_cls = int(classes[p_cpy])
            act = label_dict[label_dict["ClassId"] == act_cls]["Name"].values[0]
            prd = label_dict[label_dict["ClassId"] == prd_cls]["Name"].values[0]
            
            # Uncomment to draw pictures
            # add_margin_and_text(i_cpy, f"Actual: {act}\nPredicted: {prd}").save(f"testbatch/{act_cls}_{_}.png")

            if act not in stats:
                stats[act] = [0, 0]
            
            if act_cls == prd_cls:
                stats[act][0] += 1
            else:
                stats[act][1] += 1


            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())
            
            tq.update(1)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))

    return None

def main():
    global classes
    device = "cuda:0"
    
    test_data = TrafficSignDataset("test")
    classes = test_data.class_list
    test_loader = DataLoader(test_data, batch_size=1)
    resnet = Resnet(54).to(device)
    checkpoint = torch.load(f"checkpoints/model_final.pth")
    resnet.load_state_dict(checkpoint["state_dict"])
    test(resnet, test_loader, device)

    # Classwise accuracy
    for k in stats:
        print(k, stats[k])



if __name__ == "__main__":
    main()
