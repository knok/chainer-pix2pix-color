# -*- coding: utf-8 -*-
#

import os
import numpy as np
from PIL import Image

import chainer
from chainer.dataset import dataset_mixin

class Pix2pixDataset(dataset_mixin.DatasetMixin):
    def __init__(self, datadir):
        self.pos = 0
        self.epoch = 0
        self.datadir = datadir
        files = []
        for fname in os.listdir(datadir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                files.append(fname)
        self.files = files
        if len(files) <= 0:
            raise FileNotFoundError("no image file in the dir %s" % datadir)

    def __len__(self):
        return len(self.files)

    def load_image(self, i):
        path = os.path.join(self.datadir, self.files[i])
        img = Image.open(path)
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img /= 255.0
        return img

    # return (A image, B image)
    def get_example(self, i):
        if i > len(self):
            raise IndexError("index too large")
        img = self.load_image(i)
        ch, h, w = img.shape
        w = w // 2
        a_img = np.zeros((ch, h, w), dtype=np.float32)
        b_img = np.zeros((ch, h, w), dtype=np.float32)
        a_img = img[:, :, 0:w]
        b_img = img[:, :, w:w*2]
        return a_img, b_img

    def next(self):
        ret = self.get_example(self.pos)
        self.pos += 1
        return ret

    def reset(self):
        self.pos = 0

class Pix2pixIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.batchsize = batchsize
        self.pos = 0
        self.epoch = 0

    def fetch(self):
        data = []
        try:
            for i in range(self.batchsize):
                data.append(self.dataset.next())
            self.pos += self.batchsize
        except:
            self.pos = 0
            self.epoch += 1
            self.dataset.reset()
        return data

    def __next__(self):
        data = self.fetch()
        if len(data) <= 0:
            data.self.fetch()
        data = np.asarray(data)
        return data

    @property
    def epoch_detail(self):
        ed = self.epoch + float(self.pos / len(self.dataset))
        return ed

if __name__ == '__main__':
    try:
        x = Pix2pixDataset(".")
    except FileNotFoundError as e:
        print(e)
    #x = Pix2pixDataset("/path/to/data")
    #y = Pix2pixIterator(x, 1)
