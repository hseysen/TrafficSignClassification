import os
import tqdm
from utils import retrieve_norms
import torch
import torch.nn as nn
from torchmetrics import F1Score
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from datasets.dataset_retrieval import TrafficSignDataset
from models.ResnetModel import Resnet
from models.VGGModel import VGG16
from torch.utils.tensorboard import SummaryWriter


MODEL_PATH = "checkpoints/"
PTH_NAME = f"model_vgg16_sgd.pth"


def val(model, data_val, loss_function, writer, epoch, device):
    f1score = 0
    f1 = F1Score(num_classes=54, task="multiclass")
    data_iterator = enumerate(data_val)
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()
        tq = tqdm.tqdm(total=len(data_val))
        tq.set_description("Validation:")

        total_loss = 0
        for _, batch in data_iterator:
            # Forward propagation
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            
            # Compute loss
            loss = loss_function(pred, label.float())
            pred = pred.softmax(dim=1)
            
            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())
            
            total_loss += loss.item()
            tq.update(1)

    f1score = f1(torch.tensor(f1_list), torch.tensor(f1t_list))
    writer.add_scalar("Validation F1", f1score, epoch)
    writer.add_scalar("Validation Loss", total_loss / len(data_val), epoch)

    tq.close()
    print(f"F1 score: {f1score}")

    return None


def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device):
    writer = SummaryWriter()

    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        model.train()
        tq = tqdm.tqdm(total=len(train_loader))
        tq.set_description(f"Epoch #{epoch}")

        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st=f"{loss.item():.6f}")
            tq.update(1)
        
        writer.add_scalar(f"Training Loss", running_loss / len(train_loader), epoch)
        
        tq.close()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1} / {n_epochs}], Loss: {epoch_loss:.4f}")
        
        # Check the performance of the model on unseen dataset
        val(model, val_loader, loss_fn, writer, epoch, device)
        
        # Save the model in ".pth" format
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(MODEL_PATH, PTH_NAME))
        print(f"saved the model {epoch_loss}")


def main():
    device = "cuda:0"
    mean, stdv = retrieve_norms()
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    train_data = TrafficSignDataset("train", transforms=tr)
    val_data = TrafficSignDataset("valid", transforms=tr)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2, drop_last=True)

    # model = Resnet(54).to(device)
    model = VGG16(54).to(device)
    optimizer = SGD(model.parameters(), lr=0.001)
    # optimizer = Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    max_epoch = 20
    train(model, train_loader, val_loader, optimizer, loss, max_epoch, device)


if __name__ == "__main__":
    main()