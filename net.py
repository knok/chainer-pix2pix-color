# -*- coding: utf-8 -*-
#

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

class UNetGenerator(chainer.Chain):

    def __init__(self, ngf, **kwargs):
        self.ngf = ngf
        self.ch = 3
        self.width = 256
        self.height = 256
        super(UNetGenerator, self).__init__(**kwargs)

        with self.init_scope():
            # conv1 [batch, ch, 256, 256] -> [batch, ngf, 128, 128]
            self.conv1 = L.Convolution2D(self.ch, self.ngf, stride=2)
            layer_specs = [
                ngf * 2, # conv2 [batch, ngf*2, 64, 64]
                ngf * 4, # conv3 [batch, ngf*4, 32, 32]
                ngf * 8, # conv4 [batch, ngf*8, 16, 16]
                ngf * 8, # conv5 [batch, ngf*8, 8, 8]
                ngf * 8, # conv6 [batch, ngf*8, 4, 4]
                ngf * 8, # conv7 [batch, ngf*8, 2, 2]
                ngf * 8, # conv8 [batch, ngf*8, 1, 1]
            ]
            self.encoder_layer_specs = layer_specs
            self.encoder_num_layers = len(layer_specs) + 2
            for i, out_channels  in enumerate(layer_specs):
                cname = "conv%d" % (i + 2)
                setattr(self, cname, L.Convolution2D(None, out_channels, ksize=3, stride=2, pad=1))
                bnname = "encbn%d" % (i + 2)
                setattr(self, bnname, L.BatchNormalization(out_channels))

            layer_specs = [
                (ngf * 8, 0.5),
                (ngf * 8, 0.5),
                (ngf * 8, 0.5),
                (ngf * 8, 0.0),
                (ngf * 4, 0.0),
                (ngf * 2, 0.0),
                (ngf, 0.0),
            ]
            self.decoder_layer_specs = layer_specs
            self.decoder_num_layers = len(layer_specs) + 2
            for i in range(self.decoder_num_layers, 1, -1):
                cname = "deconv%d" % i
                bnname = "decbn%d" % i
                out_channels, _ = layer_specs[self.decoder_num_layers - i - 2]
                setattr(self, cname, L.Deconvolution2D(None, out_channels, ksize=2, stride=2, pad=1))
                setattr(self, bnname, L.BatchNormalization(out_channels))
            self.deconv1 = L.Deconvolution2D(ngf, self.ch, ksize=2, stride=2, pad=1)

    def predict(self, x):
        input = x
        layers = []
        output = self.conv1(input)
        for i in range(2, self.encoder_num_layers):
            cname = "conv%n" % i
            bnname = "encbn%n" % i
            ref = F.leaky_relu(output, 0.2)
            h = self[cname](ref)
            output = self[bnname](h)
            layers.append(output)
        input = output
        for i in range(8, 1, -1):
            idx = i - 2
            dcname = "deconv%n" % idx
            bnname = "decbn%n" % idx
            ngf, dropout = self.decoder_layer_spec[idx]
            input = F.concat([layers[idx], input])
            ref = F.relu(input)
            h = self[dcname](ref)
            output = self[bnname](h)
            if dropout > 0.0:
                output = F.dropout(output, dropout)
        input = F.concat([output, layers[-1]])
        ref = F.relu(input)
        h = self.deconv1(ref)
        output = F.tanh(h)

        return output

if __name__ == "__main__":
    net = UNetGenerator(32)
