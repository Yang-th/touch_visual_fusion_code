import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import util
import dataloader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import fusion_network
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import C3D_model
import sys
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


def main():

    ''' Load data '''
    root_train = 'face_cut/A_data'
    list_train = 'face_cut/A_dataset.txt'
    batchsize_train = 1
    root_eval = 'face_cut/B_data'
    list_eval = 'face_cut/B_dataset.txt'
    batchsize_eval = 1
    train_dataset = dataloader.Enterfacedataset(root_train, list_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize_train, shuffle=True)  # ,
    # sampler=train_sampler)
    test_dataset = dataloader.Enterfacedataset(root_eval, list_eval)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchsize_eval, shuffle=False)

    print('读取训练数据集合完成')


    num_epochs = 16
    lr = 0.001
    # declare0 and define an objet of MyCNN
    # fusion = fusion_network.Fusion()

    # fusion = fusion_network.CompactBilinearPooling(256, 256, 2048)
    net = C3D_model.C3D(num_classes=6, pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.fc6.parameters():
        param.requires_grad = True
    for param in net.fc7.parameters():
        param.requires_grad = True
    for param in net.fc8.parameters():
        param.requires_grad = True
    # print(mycnn)

    device = torch.device('cuda')

    # optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    losses, val_losses, accs, time, y_pre_one, y_ture_one, _ = fit(net, num_epochs, optimizer, device, lr,
                                                                   train_loader, test_loader, lr_scheduler)


def fit(model, num_epochs, optimizer, device, lr, train_loader, test_loader, lr_scheduler):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model.
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    val_losses = []
    accs = []
    step_index = 0
    data_times = []
    end = time.time()
    predict = []
    lab = []
    train_accuracy = []
    best_model_acc = 0
    for epoch in range(num_epochs):
        # if epoch % 10 == 0:
        #     step_index += 1
        # lr = step_lr(optimizer, lr, epoch, 0.5, step_index)
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # train step
        loss, trainacc = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)
        train_accuracy.append(trainacc)
        train_time = time.time() - end
        print(train_time)
        data_times.append(train_time)
        # evaluate step
        test_loss, accuracy, y_predict, label = evaluate(model, test_loader, loss_func, device)
        if best_model_acc < accuracy:
            best_model_acc = accuracy
            # torch.save(model.state_dict(), 'model/model.pt')
        #         print(label)
        val_losses.append(test_loss)
        accs.append(accuracy)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        end = time.time()
        predict.append(np.array(y_predict))
        lab.append(np.array(label))
        lr_scheduler.step()

    # show curve
    lab = np.array(lab)
    predict = np.array(predict)
    show_curve(losses, "train loss")
    show_curve(val_losses, "test loss")
    show_curve(train_accuracy, "train accuracy")
    show_curve(accs, "test accuracy")
    print(lab.shape)
    return losses, val_losses, accs, time, predict, lab, train_accuracy


def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()


def step_lr(optimizer, learning_rate, epoch, gamma, step_index):
    lr = learning_rate
    if (epoch % 10 == 0):  # &(epoch ==200):
        lr = learning_rate * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.
    model: CNN networks
    train_loader: a Dataloader object with training data
    loss_func: loss function
    device: train on cpu or gpu device
    """
    total_loss = 0
    correct = 0
    total = 0
    # train the model using minibatch
    for i, (img, targets) in enumerate(train_loader):
        img = img.type(torch.FloatTensor)
        img = img.to(device)

        targets = targets.to(device)

        # forward
        outputs = model(img)
        loss = loss_func(outputs, targets.long())

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, dim=1)
        correct += (predicted == targets.long()).sum().item()

        total += targets.size(0)

        # every 100 iteration, print loss
        if (i + 1) % 100 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(train_loader), loss.item()))
    accuracy = correct / total
    print("train acc: {:.4f}%"
          .format(100 * accuracy))
    return total_loss / len(train_loader), accuracy


def evaluate(model, val_loader, loss_func, device):
    """
    model: CNN networks
    val_loader: a Dataloader object with validation data
    device: evaluate on cpu or gpu device
    return classification accuracy of the model on val dataset
    """
    # evaluate the model
    model.eval()
    y_predict = []
    label = []
    feature = []
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        valid_loss = 0
        for i, (img, targets) in enumerate(val_loader):
            # device: cpu or gpu
            img = img.type(torch.FloatTensor)
            img = img.to(device)
            targets = targets.to(device)

            outputs = model(img)
            loss = loss_func(outputs, targets.long())
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)

            y_predict.extend(predicted.cpu().numpy())
            if targets.long() in predicted:
                correct += 1
            label.extend(targets.cpu().numpy())
            total += targets.size(0)

        accuracy = correct / total

        print('test Loss: {:.4f} \t Accuracy on test Set: {:.4f} %'.format(valid_loss / len(val_loader),
                                                                           100 * accuracy))
        return valid_loss / len(val_loader), accuracy, y_predict, label


def save_model(model, save_path):
    # save model
    torch.save(model.state_dict(), 'model/model.pt')


if __name__ == '__main__':
    main()

