
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from network import resnet
from utils.earlystop import EarlyStopping

def plt_line(line_list, save_path, show=True):
    plt.ion()
    plt.figure(figsize=(20, 5))
    count = len(line_list)
    for i in range(count):
        # 如果没有给数据，则显示默认值
        line_list[i]['color'] = line_list[i]['color'] if 'color' in line_list[i].keys() else 'deepskyblue'
        line_list[i]['linestyle'] = line_list[i]['linestyle'] if 'linestyle' in line_list[i].keys() else '-'
        line_list[i]['marker'] = line_list[i]['marker'] if 'marker' in line_list[i].keys() else 'o'
        line_list[i]['markersize'] = line_list[i]['markersize'] if 'markersize' in line_list[i].keys() else 2
        line_list[i]['markerfacecolor'] = line_list[i]['markerfacecolor'] if 'markerfacecolor' in line_list[i].keys() else 'blue'
        line_list[i]['markeredgecolor'] = line_list[i]['markeredgecolor'] if 'markeredgecolor' in line_list[i].keys() else 'green'
        line_list[i]['markeredgewidth'] = line_list[i]['markeredgewidth'] if 'markeredgewidth' in line_list[i].keys() else 1

        # 绘制图形
        plt.plot(line_list[i]['xdata'], line_list[i]['ydata'], color=line_list[i]['color'],
                 linestyle=line_list[i]['linestyle'],
                 linewidth=1, marker=line_list[i]['marker'], markersize=line_list[i]['markersize'],
                 markerfacecolor=line_list[i]['markerfacecolor'], markeredgecolor=line_list[i]['markeredgecolor'],
                 markeredgewidth=line_list[i]['markeredgewidth'], label=line_list[i]['label'])
    # plt.grid()
    # plt.ylim(0, 1.1)
    plt.xticks(np.arange(min(line_list[0]['xdata']), max(line_list[0]['xdata']) + 1, 1))  # 设置刻度为从最小值到最大值
    plt.legend()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def adjust_learning_rate(optimizer, min_lr=1e-6):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 10.
        if param_group['lr'] < min_lr:
            return False
    return True


def save_networks(model, epoch, opt):
    os.makedirs(opt.save_path, exist_ok=True)
    save_filename = 'model_epoch_%s.pth' % epoch
    save_path = os.path.join(opt.save_path, save_filename)
    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
    }
    torch.save(state_dict, save_path)

def validate(model, dataroot, device):
    model.eval()
    correct = 0
    loss_t = 0.0
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            pred = model(img).sigmoid()

            loss = loss_fn(pred, label)
            loss_t += loss.item()
            correct += (pred.argmax(-1) == label).type(torch.float).sum().item()

    correct /= len(dataloader.dataset)
    loss_t /= len(dataloader)
    print(f'Accuracy: {correct}')
    return correct, loss_t


args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--num_workers', type=int, default=0)
args.add_argument('--epochs', type=int, default=1000)
args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args.add_argument('--earlystop_epoch', type=int, default=5)
args.add_argument('--loss_freq', type=int, default=50)
args.add_argument('--save_latest_freq', type=int, default=1000)
args.add_argument('--name', type=str, default='monkeypoxskin')
args.add_argument('--save_path', type=str, default='checkpoints')
args.add_argument('--save_epoch_freq', type=int, default=5)
args.add_argument('--dataroot', type=str, default='H:/机器学习/leak_code(resnet50)/data/DataSet_5s_single_image')

opt = args.parse_args(args=[])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.RandomRotation(30),#随机旋转
    transforms.ToTensor(),   # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=os.path.join(opt.dataroot, 'train'), transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

# choose model
model = resnet.resnet50(pretrained=True)
#model.fc = nn.Linear(512, 2)#最后一层卷积层输出的特征图大小为512，512由ResNet 的结构和参数决定；2表示输出元素，两个元素的向量，每个元素对应输出的类别的概率，即这里2指二分类
#如果模型是resnet50，改为model.fc = nn.Linear(2048, 2)
model.fc = nn.Linear(2048, 2)
model = model.to(opt.device)

loss_fn = nn.CrossEntropyLoss().to(opt.device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

loss_train_list = []
acc_train_list = []
loss_val_list = []
acc_val_list = []
total_step = 0

for epoch in range(opt.epochs):
    model.train()
    loss_train = 0.0
    acc_train = 0.0
    for i, (img, label) in enumerate(dataloader):
        img = img.to(opt.device)
        label = label.to(opt.device)

        pred = model(img)
        loss = loss_fn(pred, label)
        loss_train += loss.item()
        pred = model(img).sigmoid()
        acc_train += (pred.argmax(-1) == label).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_step % opt.loss_freq == 0:
            print(f'Epoch [{epoch}/{opt.epochs}] Batch [{i}/{len(dataloader)}] Loss: {loss.item()}')

        if total_step % opt.save_latest_freq == 0:
            print('saving the latest model %s (epoch %d)' % (opt.name, epoch))
            save_networks(model, 'latest', opt)

        total_step += 1
    acc_train = acc_train / len(dataloader.dataset)
    val_acc, val_loss = validate(model, os.path.join(opt.dataroot, 'val'), opt.device)

    print("(Val @ epoch {}) acc: {};".format(epoch, val_acc))
    loss_train_list.append(loss_train / len(dataloader))
    acc_train_list.append(acc_train)
    loss_val_list.append(val_loss)
    acc_val_list.append(val_acc)

    plt_line([
        {'xdata': range(len(loss_train_list)), 'ydata': loss_train_list, 'label': 'loss_train'},
        {'xdata': range(len(loss_val_list)), 'ydata': loss_val_list, 'label': 'loss_val', 'color': 'red'}
    ], 'result/loss.png', show=True)

    plt_line([
        {'xdata': range(len(acc_train_list)), 'ydata': acc_train_list, 'label': 'acc_train'},
        {'xdata': range(len(acc_val_list)), 'ydata': acc_val_list, 'label': 'acc_val', 'color': 'red'}
    ], 'result/acc.png', show=True)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_step))
        save_networks(model, 'latest', opt)
        # save_networks(model, epoch, opt)
    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        cont_train = adjust_learning_rate(optimizer)
        if cont_train:
            print("Learning rate dropped by 10, continue training...")
            early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
        else:
            print("Early stopping.")
            break
    model.train()

loss_train_list = np.array(loss_train_list)
acc_train_list = np.array(acc_train_list)
loss_val_list = np.array(loss_val_list)
acc_val_list = np.array(acc_val_list)
np.save('result/loss_train.npy', loss_train_list)
np.save('result/acc_train.npy', acc_train_list)
np.save('result/loss_val.npy', loss_val_list)
np.save('result/acc_val.npy', acc_val_list)