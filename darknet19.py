import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *
import time


class ChainBuilder(object):

    def get_conv_stack(self, input, output,
                       ksize=3, stride=1, pad=1, nobias=True,
                       use_beta=False):
        conv = L.Convolution2D(input, output, ksize=ksize, stride=stride,
                               pad=pad, nobias=nobias)
        bn = L.BatchNormalization(output, use_beta=use_beta)
        bias = L.Bias(shape=(output,))

        return conv, bn, bias

    def get_activate(self, x, conv, bn, bias, finetune=True):
        return F.leaky_relu(bias(bn(conv(x), finetune=finetune)), slope=0.1)


class Darknet19(Chain, ChainBuilder):

    """
    Darknet19
    - It takes (224, 224, 3) or (448, 448, 4) sized image as input
    """

    def __init__(self):

        conv1, bn1, bias1 = self.get_conv_stack(3, 32)
        conv2, bn2, bias2 = self.get_conv_stack(32, 64)
        conv3, bn3, bias3 = self.get_conv_stack(64, 128)
        conv4, bn4, bias4 = self.get_conv_stack(128, 64, ksize=1, pad=0)
        conv5, bn5, bias5 = self.get_conv_stack(64, 128)
        conv6, bn6, bias6 = self.get_conv_stack(128, 256)
        conv7, bn7, bias7 = self.get_conv_stack(256, 128, ksize=1, pad=0)
        conv8, bn8, bias8 = self.get_conv_stack(128, 256)
        conv9, bn9, bias9 = self.get_conv_stack(256, 512)
        conv10, bn10, bias10 = self.get_conv_stack(512, 256, ksize=1, pad=0)
        conv11, bn11, bias11 = self.get_conv_stack(256, 512)
        conv12, bn12, bias12 = self.get_conv_stack(512, 256, ksize=1, pad=0)
        conv13, bn13, bias13 = self.get_conv_stack(256, 512)
        conv14, bn14, bias14 = self.get_conv_stack(512, 1024)
        conv15, bn15, bias15 = self.get_conv_stack(1024, 512, ksize=1, pad=0)
        conv16, bn16, bias16 = self.get_conv_stack(512, 1024)
        conv17, bn17, bias17 = self.get_conv_stack(1024, 512, ksize=1, pad=0)
        conv18, bn18, bias18 = self.get_conv_stack(512, 1024)

        super(Darknet19, self).__init__(
            ##### common layers for both pretrained layers and yolov2 #####
            conv1=conv1, bn1=bn1, bias1=bias1,
            conv2=conv2, bn2=bn2, bias2=bias2,
            conv3=conv3, bn3=bn3, bias3=bias3,
            conv4=conv4, bn4=bn4, bias4=bias4,
            conv5=conv5, bn5=bn5, bias5=bias5,
            conv6=conv6, bn6=bn6, bias6=bias6,
            conv7=conv7, bn7=bn7, bias7=bias7,
            conv8=conv8, bn8=bn8, bias8=bias8,
            conv9=conv9, bn9=bn9, bias9=bias9,
            conv10=conv10, bn10=bn10, bias10=bias10,
            conv11=conv11, bn11=bn11, bias11=bias11,
            conv12=conv12, bn12=bn12, bias12=bias12,
            conv13=conv13, bn13=bn13, bias13=bias13,
            conv14=conv14, bn14=bn14, bias14=bias14,
            conv15=conv15, bn15=bn15, bias15=bias15,
            conv16=conv16, bn16=bn16, bias16=bias16,
            conv17=conv17, bn17=bn17, bias17=bias17,
            conv18=conv18, bn18=bn18, bias18=bias18,

            # new layer
            conv19=L.Convolution2D(1024, 10, ksize=1, stride=1, pad=0),
        )
        self.train = False
        self.finetune = False

    def __call__(self, x):
        batch_size = x.data.shape[0]

        # common layer
        h = self.get_activate(x, self.conv1, self.bn1, self.bias1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = self.get_activate(h, self.conv2, self.bn2, self.bias2)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = self.get_activate(h, self.conv3, self.bn3, self.bias3)
        h = self.get_activate(h, self.conv4, self.bn4, self.bias4)
        h = self.get_activate(h, self.conv5, self.bn5, self.bias5)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = self.get_activate(h, self.conv6, self.bn6, self.bias6)
        h = self.get_activate(h, self.conv7, self.bn7, self.bias7)
        h = self.get_activate(h, self.conv8, self.bn8, self.bias8)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = self.get_activate(h, self.conv9, self.bn9, self.bias9)
        h = self.get_activate(h, self.conv10, self.bn10, self.bias10)
        h = self.get_activate(h, self.conv11, self.bn11, self.bias11)
        h = self.get_activate(h, self.conv12, self.bn12, self.bias12)
        h = self.get_activate(h, self.conv13, self.bn13, self.bias13)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = self.get_activate(h, self.conv14, self.bn14, self.bias14)
        h = self.get_activate(h, self.conv15, self.bn15, self.bias15)
        h = self.get_activate(h, self.conv16, self.bn16, self.bias16)
        h = self.get_activate(h, self.conv17, self.bn17, self.bias17)
        h = self.get_activate(h, self.conv18, self.bn18, self.bias18)

        # new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape
        y = F.reshape(h, (batch_size, -1))
        return y


class Darknet19Predictor(Chain):
    def __init__(self, predictor):
        super(Darknet19Predictor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)

        if t.ndim == 2:  # use squared error when label is one hot label
            y = F.softmax(y)
            # loss = F.mean_squared_error(y, t)
            loss = sum_of_squared_error(y, t)
            accuracy = F.accuracy(y, t.data.argmax(axis=1).astype(np.int32))
        else:  # use softmax cross entropy when label is normal label
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)

        return y, loss, accuracy

    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)
