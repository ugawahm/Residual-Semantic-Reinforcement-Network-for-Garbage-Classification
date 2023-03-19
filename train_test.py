import os
import sys
import json
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
from matplotlib import pyplot as plt
# from prettytable import PrettyTable
# from torch.optim import lr_scheduler

# import timm
# import timm.optim
# import timm.scheduler

from torchvision import transforms, datasets
from tqdm import tqdm
from VCRNet_CBAM_1_19 import resnext50_32x4d
# from test_train import *
matplotlib.use('TkAgg')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                    # transforms.RandomRotation(30, expand=False, center=None, fill=None),
                                    # 随机旋转
                                    # transforms.RandomGrayscale(p=0.1),  # 0.1概率变成灰度
                                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),#随机光学变换
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "laji_data")
    image_path = os.path.join(data_root, "PycharmProjects", "laji12", "garbage_classification")  # laji data set path
    # image_path = os.path.join(data_root, "PycharmProjects", "laji12", "garbage_new")  # laji data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    laji_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in laji_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net = Resnext(50, 5)
    net = resnext50_32x4d(num_classes=12)  # 为迁移学习时（）内为空
    # --------------------------迁移学习分割线----------------------------------
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # 使用迁移学习则启用下面代码
    #
    # net = resnext50_32x4d()
    # model_weight_path = "weight_document/resnext50_32x4d-7cdf4587.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)    # , strict=False
    # # for param in net.parameters():
    # #     param.requires_grad = False.0
    #
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 12)                   # 分类数

    # -------------------------迁移学习分割线-----------------------------------
    net.to(device)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    # lrs = []

    # （余弦模拟退火 + warm up 动态调整学习率） + SGD优化器
    # optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0001)
    # scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
    #                                              t_initial=150,
    #                                              lr_min=0.0001,
    #                                              warmup_t=4,
    #                                              warmup_lr_init=0.0001)

    Loss_list = []
    Accuracy_list = []

    epochs = 150
    best_acc = 0.0
    save_path = 'weight_document/vcr_0318_12.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # lrs.append(optimizer.param_groups[0]["lr"])


            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # print("epoch: %d, lr：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # scheduler.step(epoch)  # 动态更新学习率

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_losses = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_losses += val_loss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num

        Loss_list.append(val_losses / val_steps)
        Accuracy_list.append(100 * val_accurate)

        print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_losses / val_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('best_acc: %.3f' % best_acc)
    # print('best_acc_total: %.3f' % best_acc)

    # 验证集loss和acc曲线可视化
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.figure(figsize=(19.2, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    # plt.title('Val accuracy vs. epochs')
    # plt.ylabel('Val accuracy')
    plt.title('验证集准确率和损失曲线')
    plt.ylabel('验证准确率')
    plt.xlabel('迭代次数')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    # plt.xlabel('Val loss vs. epochs')
    # plt.ylabel('Val loss')
    plt.xlabel('迭代次数')
    plt.ylabel('验证损失')
    plt.savefig('./results/vcr318_p12.png', dpi=300)
    plt.show()

    print('Finished Training')


if __name__ == '__main__':
    main()

