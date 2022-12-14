import einops
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import time
from torch import optim

import time
import matplotlib.pyplot as plt

from ViT import ViT

from PyUtils.pytorch.utils import write_graph
import os, sys



########################################################################################################################
# https://github.com/lordrebel/mnist-vit
########################################################################################################################


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    train_start = time.time()
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + '  AVG-Loss: ' +
                  '{:6.4f}'.format(sum(loss_history)/len(loss_history)))
    print(f'AVG-Loss: {sum(loss_history)/len(loss_history): 6.4f}  elapse-time: {time.time()-train_start} s')


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    test_start = time.time()
    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)  elapse-time: {:6.4} s'.format(time.time()-test_start) + '\n')
    torch.save(model, f'./last.pth')


def save_fig(train_loss_history, test_loss_history, train_path, test_path):
    x1 = list(range(1, len(train_loss_history)+1))
    x2 = list(range(1, len(test_loss_history)+1))
    plt.xlabel('')
    plt.ylabel('loss')
    plt.plot(x1, train_loss_history, 'r', label='train_loss')
    # plt.show()
    plt.savefig(train_path)

    plt.clf()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x2, test_loss_history, 'b', label='test_loss')
    # plt.show()
    plt.savefig(test_path)


def main(CONFIGS):
    torch.manual_seed(42)
    DOWNLOAD_PATH = './data'

    N_EPOCHS = CONFIGS['N_EPOCHS']
    BATCH_SIZE_TRAIN = CONFIGS['BATCH_SIZE_TRAIN']
    BATCH_SIZE_TEST = CONFIGS['BATCH_SIZE_TEST']
    img_size = CONFIGS['IMG_SIZE']
    patch_size = CONFIGS['PATCH_SIZE']
    num_classes = CONFIGS['NUM_CLASSES']
    channels = CONFIGS['CHANNELS']
    dim = CONFIGS['DIM']
    heads = CONFIGS['MULTI_HEADS']
    depth = CONFIGS['DEPTH']
    mlp_dim = CONFIGS['MLP_DIM']

    transform_mnist = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                           transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                          transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    start_time = time.time()
    '''
    patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
    在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。'''
    model = ViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes, channels=channels,
                dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
    # model.eval()
    # write_graph(model, torch.randn(2, 1, img_size, img_size), './model_graph.pdf', True)

    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)
    save_fig(train_loss_history, test_loss_history, './tmp/train_loss.png', './tmp/test_loss.png')

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')



def TEST_patch_size():
    CONFIGS = {
        'IMG_SIZE': 64,
        'PATCH_SIZE': 4,
        'NUM_CLASSES': 10,
        'CHANNELS': 1,
        'DIM': 64,
        'MULTI_HEADS': 4,
        'DEPTH': 2,
        'MLP_DIM': 128,
        'BATCH_SIZE_TRAIN': 64,
        'BATCH_SIZE_TEST': 128,
        'N_EPOCHS': 10
    }
    main(CONFIGS)


if __name__ == '__main__':

    TEST_patch_size()