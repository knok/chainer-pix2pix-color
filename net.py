# -*- coding: utf-8 -*-
#

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample="down", activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class UNetGenerator(chainer.Chain):

    def __init__(self, ngf, **kwargs):
        self.ngf = ngf
        self.ch = 3
        self.width = 256
        self.height = 256
        super(UNetGenerator, self).__init__(**kwargs)
        
        with self.init_scope():
            layers = {}
            # conv1 [batch, ch, 256, 256] -> [batch, ngf, 128, 128]
            layers['c0'] = L.Convolution2D(self.ch, self.ngf, 3, 1, 1)
            layers['c1'] = CBR(self.ngf, self.ngf*2, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c2'] = CBR(self.ngf*2, self.ngf*4, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c3'] = CBR(self.ngf*4, self.ngf*8, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c4'] = CBR(self.ngf*8, self.ngf*8, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c5'] = CBR(self.ngf*8, self.ngf*8, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c6'] = CBR(self.ngf*8, self.ngf*8, bn=True, sample="down",
                               activation=F.leaky_relu, dropout=False)
            layers['c7'] = CBR(self.ngf*8, self.ngf*8, bn=True, sample="down",
                                activation=F.leaky_relu, dropout=False)
            #
            # Decoder
            layers['d0'] = CBR(self.ngf*8, self.ngf*8, bn=True, sample="up",
                               activation=F.relu, dropout=True)
            layers['d1'] = CBR(self.ngf*8*2, self.ngf*8, bn=True, sample="up",
                               activation=F.relu, dropout=True)
            layers['d2'] = CBR(self.ngf*8*2, self.ngf*8, bn=True, sample="up",
                               activation=F.relu, dropout=True)
            layers['d3'] = CBR(self.ngf*8*2, self.ngf*8, bn=True, sample="up",
                               activation=F.relu, dropout=False)
            layers['d4'] = CBR(self.ngf*8*2, self.ngf*4, bn=True, sample="up",
                               activation=F.relu, dropout=False)
            layers['d5'] = CBR(self.ngf*8, self.ngf*2, bn=True, sample="up",
                               activation=F.relu, dropout=False)
            layers['d6'] = CBR(self.ngf*4, self.ngf, bn=True, sample="up",
                               activation=F.relu, dropout=False)
            w = chainer.initializers.Normal(0.02)
            layers['d7'] = L.Convolution2D(self.ngf*2, self.ch, 3, 1, 1, initialW=w)
            for k, v in layers.items():
                setattr(self, k, v)

    def predict(self, x):
        input = x
        hs = [F.leaky_relu(self.c0(input))]
        for i in range(1, 8):
            h = self['c%d'%i](hs[-1])
            hs.append(h)
        h = self.d0(hs[-1])
        for i in range(1, 8):
            h = F.concat([h, hs[-i-1]])
            if i < 7:
                h = self['d%d'%i](h)
            else:
                h = self.d7(h)
        output = F.tanh(h)
        return output

    def __call__(self, x):
        return self.predict(x)

class P2PDiscriminator(chainer.Chain):
    def __init__(self, ndf, **kwargs):
        super(P2PDiscriminator, self).__init__(**kwargs)
        self.ch = 3
        self.ndf = ndf
        self.n_layers = 3
        layers = {}
        with self.init_scope():
            layers['c0_0'] = CBR(self.ch, self.ndf, bn=False, sample='down',
                                 activation=F.leaky_relu, dropout=False)
            layers['c0_1'] = CBR(self.ch, self.ndf, bn=False, sample='down',
                                 activation=F.leaky_relu, dropout=False)
            layers['c1'] = CBR(self.ndf*2, self.ndf*4, bn=False, sample='down',
                                 activation=F.leaky_relu, dropout=False)
            layers['c2'] = CBR(self.ndf*4, self.ndf*8, bn=False, sample='down',
                                 activation=F.leaky_relu, dropout=False)
            layers['c3'] = CBR(self.ndf*8, self.ndf*16, bn=False, sample='down',
                                 activation=F.leaky_relu, dropout=False)
            w = chainer.initializers.Normal(0.02)
            layers['c4'] = L.Convolution2D(self.ndf*16, 1, 3, 1, 1, initialW=w)
            for k, v in layers.items():
                setattr(self, k, v)

    def __call__(self, x0, x1):
        h = F.concat([self.c0_0(x0), self.c0_1(x1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        output = F.sigmoid(h)
        return output

if __name__ == "__main__":
    net = UNetGenerator(32)
    disc = P2PDiscriminator(32)
