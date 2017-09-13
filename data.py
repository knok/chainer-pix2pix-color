# -*- coding: utf-8 -*-
#

import os
import numpy as np
from PIL import Image

from chainer.dataset import DatasetMixin

class Pix2pixDataset(DatasetMixin):
    def __init__(self, datadir):
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
        img = Image.load(self.files[i])
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img /= 255.0
        return img

    # return (A image, B image)
    def get_example(self, i):
        img = self.load_image(i)
        ch, w, h = img.shape
        w = w // 2
        h = h // 2
        a_img = np.zeros((ch, w, h), dtype=np.float32)
        b_img = np.zeros((ch, w, h), dtype=np.float32)
        a_img = img[:, 0:w, 0:h]
        b_img = img[:, w:w*2, h:h*2]
        return a_img, b_img

if __name__ == '__main__':
    try:
        x = Pix2pixDataset(".")
    except FileNotFoundError as e:
        print(e)
