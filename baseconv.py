# conv
from PB_CONV.conv_acc_lib.layers_2 import ConvolutionalLayer
# import paibox as pb
import numpy  as np


def conv_trans_fc(conv_weight_np,INPUT_SIZE,STRIDE):
    
    INPUT_CN = conv_weight_np.shape[0]
    KERNEL_HW = conv_weight_np.shape[1]
    OUTPUT_CN = conv_weight_np.shape[3]

    INPUT_SIZE_H = INPUT_SIZE[0]
    INPUT_SIZE_W = INPUT_SIZE[1]
 
    OUTPUT_SIZE_H = int((INPUT_SIZE_H - KERNEL_HW)/STRIDE) + 1
    OUTPUT_SIZE_W = int((INPUT_SIZE_W - KERNEL_HW)/STRIDE) + 1

    conv2fc_weight = np.zeros([INPUT_CN*INPUT_SIZE_H*INPUT_SIZE_W,OUTPUT_CN*OUTPUT_SIZE_H*OUTPUT_SIZE_W])
    zero_image     = np.zeros([INPUT_CN*INPUT_SIZE_H,INPUT_SIZE_W*OUTPUT_CN,OUTPUT_SIZE_H*OUTPUT_SIZE_W])

    for i in range(0,OUTPUT_SIZE_H):
        for j in range(0,OUTPUT_SIZE_W):
            for chi in range(INPUT_CN):
                for cho in range(OUTPUT_CN):
                    zero_image[(i*STRIDE+chi*INPUT_SIZE_H):(i*STRIDE+chi*INPUT_SIZE_H)+KERNEL_HW,(j*STRIDE+cho*INPUT_SIZE_W):(j*STRIDE+cho*INPUT_SIZE_W)+KERNEL_HW,i*OUTPUT_SIZE_W+j] = conv_weight_np[chi,:,:,cho]
            t = zero_image[:,:,i*OUTPUT_SIZE_W+j].reshape([INPUT_CN*INPUT_SIZE_H,OUTPUT_CN,INPUT_SIZE_W]).transpose(1,0,2)
            for cho in range(OUTPUT_CN):
                conv2fc_weight[:,i*OUTPUT_SIZE_W+j+cho*OUTPUT_SIZE_W*OUTPUT_SIZE_H] = t[cho].flatten()

    return conv2fc_weight

def pooling_trans_fc(INPUT_CN,INPUT_SIZE,KERNEL_HW,STRIDE):
    INPUT_SIZE_H = INPUT_SIZE[0]
    INPUT_SIZE_W = INPUT_SIZE[1]
    OUTPUT_CN    = INPUT_CN
    OUTPUT_SIZE_H = int((INPUT_SIZE_H - KERNEL_HW)/STRIDE) + 1
    OUTPUT_SIZE_W = int((INPUT_SIZE_W - KERNEL_HW)/STRIDE) + 1
    pooling2fc_weight = np.zeros([INPUT_CN*INPUT_SIZE_H*INPUT_SIZE_W,OUTPUT_CN*OUTPUT_SIZE_H*OUTPUT_SIZE_W])
    zero_image     = np.zeros([INPUT_CN*INPUT_SIZE_H,INPUT_CN*INPUT_SIZE_W])
    pooling_weight = np.ones([KERNEL_HW,KERNEL_HW])
    for i in range(0,OUTPUT_SIZE_H):
        for j in range(0,OUTPUT_SIZE_W):
            zero_image     = np.zeros([INPUT_CN*INPUT_SIZE_H,INPUT_CN*INPUT_SIZE_W])
            for chi in range(INPUT_CN):
                zero_image[(i*STRIDE+chi*INPUT_SIZE_H):(i*STRIDE+chi*INPUT_SIZE_H)+KERNEL_HW,(j*STRIDE+chi*INPUT_SIZE_W):(j*STRIDE+chi*INPUT_SIZE_W)+KERNEL_HW] = pooling_weight
            temp = zero_image.reshape([INPUT_CN*INPUT_SIZE_H,OUTPUT_CN,INPUT_SIZE_W]).transpose(1,0,2)
            for cho in range(OUTPUT_CN):
                pooling2fc_weight[:,i*OUTPUT_SIZE_W+j+cho*OUTPUT_SIZE_W*OUTPUT_SIZE_H] = temp[cho].flatten()
    return pooling2fc_weight

            
    



def main():
    INPUT_CN    = 1
    KERNEL_HW   = 3
    INPUT_SIZE  = [30,30]
    INPUT_SIZE_H = INPUT_SIZE[0]
    INPUT_SIZE_W = INPUT_SIZE[1]


    OUTPUT1_CN   = 8
    STRIDE = 1
    OUTPUT1_SIZE_H = int((INPUT_SIZE_H - KERNEL_HW)/STRIDE) + 1
    OUTPUT1_SIZE_W = int((INPUT_SIZE_W - KERNEL_HW)/STRIDE) + 1

    np.random.seed(0)
    # conv_bias_np   = np.random.randint(255,size=[1])
    conv1_bias_np   = np.zeros([OUTPUT1_CN])
    conv1_weight_np = np.random.randint(255,size=[INPUT_CN,KERNEL_HW,KERNEL_HW,OUTPUT1_CN])
    conv1_layer = ConvolutionalLayer(KERNEL_HW, INPUT_CN, OUTPUT1_CN, 0, STRIDE, 1)  #k,ic,oc.paddding,stride,speedUP
    conv1_layer.init_param()
    conv1_layer.load_param(conv1_weight_np, conv1_bias_np)
    input_data = np.random.randint(255,size=[1,INPUT_CN,INPUT_SIZE_H,INPUT_SIZE_W])
    conv1_out = conv1_layer.forward(input_data)

    conv2fc1_weight = conv_trans_fc(conv1_weight_np,INPUT_SIZE,STRIDE=STRIDE)
    conv2fc1_out = np.dot(input_data.flatten(),conv2fc1_weight)
    print(conv2fc1_out.shape)
    print(conv1_out.size)
    print((conv2fc1_out.reshape(1,OUTPUT1_CN,OUTPUT1_SIZE_H,OUTPUT1_SIZE_W) == conv1_out).all())
    INPUT_CN    = 1
    KERNEL_HW   = 3
    INPUT_SIZE  = [30,30]
    INPUT_SIZE_H = INPUT_SIZE[0]
    INPUT_SIZE_W = INPUT_SIZE[1]


    OUTPUT1_CN   = 8
    STRIDE = 1

    x = pooling_trans_fc(INPUT_CN,INPUT_SIZE,KERNEL_HW,STRIDE)
    print(x.shape)
    print(x)
    input = np.ones(INPUT_CN*INPUT_SIZE_H*INPUT_SIZE_W)
    print(np.dot(input,x).shape)








    

if __name__ == '__main__':
    main()









