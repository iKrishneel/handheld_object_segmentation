#!/usr/bin/env python

import os
import sys

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, weight='', bias=''):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, name=str(weight)), dict(lr_mult=2, name=str(bias))],
                         weight_filler=dict(type='xavier', std=0.03),
                         bias_filler=dict(type='constant', value=0.2))
    return conv, L.ReLU(conv, in_place=True)

def conv_relu2(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.03),
                         bias_filler=dict(type='constant', value=0.2))
    return conv, L.ReLU(conv, in_place=True)

def deconv(bottom, nout, ks=3, stride=1):
    de_conv = L.Deconvolution(bottom,
                              convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride,
                                                     bias_term=False), param=[dict(lr_mult=0)])
    return de_conv

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fully_connected(bottom, nout, weight, bias):
    fc = L.InnerProduct(bottom, num_output=nout,
                        param=[dict(lr_mult=1, name=str(weight)), dict(lr_mult=2, name=str(bias))],
                        weight_filler=dict(type='gaussian', std=0.01),
                        bias_filler=dict(type='constant', value=0.0))
    return fc, L.ReLU(fc, in_place=True)
    
def dropout(bottom, dratio):
    return L.Dropout(bottom, dropout_ratio=dratio, in_place=True)
    
def fcn(training_data1, training_data2, batch_size=32):
    n = caffe.NetSpec()

    n.data1, n.label1 = L.Data(source=training_data1,
                               backend = P.Data.LMDB,
                               batch_size = batch_size,
                               ntop = 2,
                               include=dict(phase=0))
    n.data2, n.label2 = L.Data(source=training_data2,
                        backend = P.Data.LMDB,
                        batch_size = batch_size,
                        ntop = 2,
                    include=dict(phase=0))
    n.silence_data = L.Silence(n.label2, ntop=0)
    
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data2, 64, pad=100, weight='conv1_1_w', bias='conv1_1_b')
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64,  weight='conv1_2_w', bias='conv1_2_b')
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, weight='conv2_1_w', bias='conv2_1_b')
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, weight='conv2_2_w', bias='conv2_2_b')
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, weight='conv3_1_w', bias='conv3_1_b')
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, weight='conv3_2_w', bias='conv3_2_b')
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, weight='conv3_3_w', bias='conv3_3_b')
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, weight='conv4_1_w', bias='conv4_1_b')
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, weight='conv4_2_w', bias='conv4_2_b')
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, weight='conv4_3_w', bias='conv4_3_b')
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, weight='conv5_1_w', bias='conv5_1_b')
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,  weight='conv5_2_w', bias='conv5_2_b')
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, weight='conv5_3_w', bias='conv5_3_b')
    n.pool5 = max_pool(n.relu5_3)

    n.fc6, n.relu6 = conv_relu2(n.pool5, 4096, 7, 1, 0)
    n.fc7, n.relu7 = conv_relu2(n.fc6, 4096, 1, 1, 0)
    # n.drop7 = dropout(n.fc7, 0.5)
    
    # ## symmetrical version
    n.conv1_1_p, n.relu1_1_p = conv_relu(n.data2, 64, pad=100, weight='conv1_1_w', bias='conv1_1_b')
    n.conv1_2_p, n.relu1_2_p = conv_relu(n.relu1_1_p, 64, weight='conv1_2_w', bias='conv1_2_b')
    n.pool1_p = max_pool(n.relu1_2_p)

    n.conv2_1_p, n.relu2_1_p = conv_relu(n.pool1_p, 128,  weight='conv2_1_w', bias='conv2_1_b')
    n.conv2_2_p, n.relu2_2_p = conv_relu(n.relu2_1_p, 128, weight='conv2_2_w', bias='conv2_2_b')
    n.pool2_p = max_pool(n.relu2_2_p)

    n.conv3_1_p, n.relu3_1_p = conv_relu(n.pool2_p, 256, weight='conv3_1_w', bias='conv3_1_b')
    n.conv3_2_p, n.relu3_2_p = conv_relu(n.relu3_1_p, 256, weight='conv3_2_w', bias='conv3_2_b')
    n.conv3_3_p, n.relu3_3_p = conv_relu(n.relu3_2_p, 256, weight='conv3_3_w', bias='conv3_3_b')
    n.pool3_p = max_pool(n.relu3_3_p)

    n.conv4_1_p, n.relu4_1_p = conv_relu(n.pool3_p, 512, weight='conv4_1_w', bias='conv4_1_b')
    n.conv4_2_p, n.relu4_2_p = conv_relu(n.relu4_1_p, 512,  weight='conv4_2_w', bias='conv4_2_b')
    n.conv4_3_p, n.relu4_3_p = conv_relu(n.relu4_2_p, 512, weight='conv4_3_w', bias='conv4_3_b')
    n.pool4_p = max_pool(n.relu4_3_p)

    n.conv5_1_p, n.relu5_1_p = conv_relu(n.pool4_p, 512, weight='conv5_1_w', bias='conv5_1_b')
    n.conv5_2_p, n.relu5_2_p = conv_relu(n.relu5_1_p, 512, weight='conv5_2_w', bias='conv5_2_b')
    n.conv5_3_p, n.relu5_3_p = conv_relu(n.relu5_2_p, 512, weight='conv5_3_w', bias='conv5_3_b')
    n.pool5_p = max_pool(n.relu5_3_p)

    n.fc6_p, n.relu6_p = conv_relu2(n.pool5_p, 4096, 7, 1, 0)
    n.fc7_p, n.relu7_p = conv_relu2(n.fc6_p, 4096, 1, 1, 0)

    # n.drop7_p = dropout(n.fc7_p, 0.5)
    
    n.concat = L.Concat(n.fc7, n.fc7_p)
    n.drop7 = dropout(n.concat, 0.5)
    n.score_fr, _ = conv_relu2(n.drop7, 1, 1, 1, 0)

    n.upscore = deconv(n.score_fr, 1, 64, 32)
    #n.score = crop(n.upscore, n.data)
    
    #n.loss = L.ContrastiveLoss(n.fc7, n.fc7_p, n.label1, margin=1)
    
    return n.to_proto()

def make_net(write_path, ):
    with open(str(write_path), 'w') as f:
        f.write(str(fcn('train_1', 'train_2')))

    # with open('val.prototxt', 'w') as f:
    #     f.write(str(fcn('seg11valid')))

def main(argv):
    make_net(argv[1])
    
if __name__ == '__main__':
    main(sys.argv)
