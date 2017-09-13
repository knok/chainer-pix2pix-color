# -*- coding: utf-8 -*-
#

import os
import numpy as np
import argparse
import chainer

from data import Pix2pixDataset
from net import UNetGenerator, P2PDiscriminator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    unet = UNetGenerator(args.ngf)
    disc = P2PDiscriminator(args.ndf)
    train_data = Pix2pixDataset(args.input_dir)

if __name__ == '__main__':
    main()
