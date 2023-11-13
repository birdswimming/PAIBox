#poisson code
from typing import Optional, Type
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import paibox as pb
import numpy as np
from paibox.base import DynamicSys
# 设置随机阈值，将输入与其进行比较，得到结果，实现泊松编码
# 问题：从模拟角度，无法直接设置一个神经元组，每个神经元需要单独设置，但芯片中应该没有这个问题。
#      
def poisson(img):
    N1 = [[pb.neuron.IF(shape=1,threshold=50+int(np.random.randint(-50, 50, 1)),reset_v=0) for _ in range(28)] for _ in range(28)]
    O1 = np.zeros([28,28])
    for i in range(28):
        for j in range(28):
            O1[i,j] = N1[i][j](img[i,j])
    return O1



if __name__ == '__main__':
    data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    batch_size = 32
    # 读取测试数据，train=True读取训练数据；train=False读取测试数据
    train_dataset = datasets.MNIST(root='./test/data', train=True, transform=data_tf)
    test_dataset = datasets.MNIST(root='./test/data', train=False, transform=data_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    examples = enumerate(test_loader)
    batch_idx, (imgs, labels) = next(examples)
    img = imgs[2].numpy()
    img = img.reshape(28,28)
    np.savetxt('pic1.txt',img,fmt='%d')
    img = img * 100
    print(img)
    output = poisson(img)
    print(output)
    np.savetxt('pic2.txt',output,fmt='%d')













