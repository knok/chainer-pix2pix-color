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
            print(h.shape)
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
        self.ndf = ndf
        self.n_layers = 3
        self.layers = []
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ndf, stride=2)
            for i in range(self.n_layers):
                idx = i + 2
                cname = "conv%d" % idx
                out_channels = ndf * min(2 ** (i+1), 8)
                stride = 1 if i == (self.n_layers+2) else 2 # stride=1 on the last layer
                setattr(self, cname, L.Convolution2D(None, out_channels, stride=stride))
                bnname = "bn%d" % idx
                setattr(self, bnname, L.BatchNormalization(out_channels))
            idx = self.n_layers + 2
            cname = "conv%d" % idx
            setattr(self, cname, L.Convolution2D(None, 1, stride=1))

    def __call__(self, x0, x1):
        h = F.concat([x0, x1])
        h = self.conv1(h)
        input = F.leaky_relu(h)
        for i in range(self.n_layers):
            idx = i + 2
            cname = "conv%d" % idx
            bnname = "bn%d" % idx
            h = self[cname](input)
            h = self[bnname](h)
            output = F.leaky_relu(h)
            input = output
        idx = self.n_layers + 2
        cname = "conv%d" % idx
        h = self[cname](input)
        output = F.sigmoid(h)

        return output

if __name__ == "__main__":
    net = UNetGenerator(32)
    disc = P2PDiscriminator(32)
