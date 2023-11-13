import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import paibox as pb
import numpy as np
import baseconv as bs

class fcnet(pb.DynSysGroup):
    """两层全连接网络，用于识别mnist数据集"""
    def __init__(self,weight1,weight2):
        super().__init__()
        self.n1 = pb.InputProj(input = None,shape_out=784) 
        self.n2 = pb.neuron.IF(128,threshold=128,reset_v=0)
        self.n3 = pb.neuron.IF(10 ,threshold=128,reset_v=0)
        self.l1 = pb.synapses.NoDecay(self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All,weights=weight1)
        self.l2 = pb.synapses.NoDecay(self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All,weights=weight2)
        
        self.p1 = pb.simulator.Probe(self.l1,'output') 
        self.p3 = pb.simulator.Probe(self.n3,'output') 

class conv_mp_fcnet(pb.DynSysGroup):
    """卷积+池化+两层全连接网络，用于识别mnist数据集"""
    def __init__(self,weight1,weight2,weight3,weight4):
        super().__init__()
        self.n1 = pb.InputProj(input = None,shape_out=784) 
        self.n2 = pb.neuron.IF(2704,threshold=128,reset_v=0)
        self.n3 = pb.neuron.IF(676 ,threshold=1,reset_v=0)
        self.n4 = pb.neuron.IF(128,threshold=128,reset_v=0)
        self.n5 = pb.neuron.IF(10 ,threshold=128,reset_v=0)
        self.l1 = pb.synapses.NoDecay(self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All,weights=weight1)
        self.l2 = pb.synapses.NoDecay(self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All,weights=weight2)
        self.l3 = pb.synapses.NoDecay(self.n3, self.n4, conn_type=pb.synapses.ConnType.All2All,weights=weight3)
        self.l4 = pb.synapses.NoDecay(self.n4, self.n5, conn_type=pb.synapses.ConnType.All2All,weights=weight4)
    
        self.p3 = pb.simulator.Probe(self.n5,'output') 





def test_fc1_fd2():
    # 首先读入权重
    w1 = np.loadtxt('./model_utils/fc1_fc2/fc1.weightparameter.txt')
    w1 = w1.astype(np.int8)
    w2 = np.loadtxt('./model_utils/fc1_fc2/fc2.weightparameter.txt')
    w2 = w2.astype(np.int8)
    # 读入图片
    input_7 = np.loadtxt('./model_utils/7.txt').reshape(784)
    net = fcnet(weight1=w1.T,weight2=w2.T)
    net.n1.input = input_7
    sim = pb.Simulator(net)
    sim.run(10)
    output = np.sum(sim.data[net.p3],axis=0)
    print(np.argmax(output) == 7)


def test_conv_mp_fc1_fc2():
    # 首先读入权重
    w1 = np.loadtxt('./model_utils/conv_mp_fc1_fc2/conv1.weightparameter.txt')
    w2 = bs.pooling_trans_fc(INPUT_CN=4,INPUT_SIZE=[26,26],KERNEL_HW=2,STRIDE=2)
    w3 = np.loadtxt('./model_utils/conv_mp_fc1_fc2/fc1.weightparameter.txt')
    w4 = np.loadtxt('./model_utils/conv_mp_fc1_fc2/fc2.weightparameter.txt')
    
    # 读入图片
    input_7 = np.loadtxt('./model_utils/7.txt').reshape(784)
    # 设置网络，仿真运行
    net = conv_mp_fcnet(weight1=w1,weight2=w2,weight3=w3.T,weight4=w4.T)
    net.n1.input = input_7
    sim = pb.Simulator(net)
    sim.run(10)
    output = np.sum(sim.data[net.p3],axis=0)
    print(np.argmax(output) == 7)
    
if __name__ == '__main__':
    test_fc1_fd2()
    test_conv_mp_fc1_fc2()