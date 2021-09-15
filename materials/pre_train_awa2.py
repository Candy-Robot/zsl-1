from operator import mod
from torch.utils.data import dataloader
from awa2_dataset import AnimalDataset
from utils import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataloader
from transform_pre_matrix_binary import *
from torch.utils.tensorboard import SummaryWriter


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    writer = SummaryWriter(log_dir="./log/zsl_graph/")
    model.train()
    train_dataset = AnimalDataset(TRAIN_CLASS_PATH, transform=train_transformer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12)
    total_steps = len(train_dataloader)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        for i, (imgs, img_predicates, img_name, img_classes) in enumerate(train_dataloader):
            # img_classes 的标签是在50个类中的标签，而不是单独的seen类的标签
            imgs = CUDA(imgs)
            # 将标签转化前40个可见类的标签
            only_train_index = train_index_40()
            train_img_classes = []
            # _, transform_all_classes = transfor_matrix_binary()
            # index_to_class = mapping_index_to_class()
            for j in img_classes.tolist():
                train_img_classes.append(only_train_index[j])
            train_img_classes = torch.tensor(train_img_classes)

            optimizer.zero_grad()

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, train_img_classes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == train_img_classes.data)
            if i % 50 == 0:
                # print(
                #     "Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}".format(
                #         epoch + 1, num_epochs, i + 1, total_steps, loss.item()
                #     )
                # )
                writer.add_scalar("Loss", loss.item(), epoch * total_steps + i)
                sys.stdout.flush()

        epoch_loss = running_loss / total_steps
        epoch_acc = running_corrects.double() / total_steps
        print('Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))



    torch.save(model_conv.state_dict(), "models/{}".format("resnet50-fc-model.bin"))




        
if __name__ == "__main__":
    model_conv = get_res50_model()

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 40)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
